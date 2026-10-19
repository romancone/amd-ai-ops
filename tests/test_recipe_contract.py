from pathlib import Path
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
RECIPE = (ROOT / 'rocm.md').read_text()


def step_shell(step_id):
    section = re.split(r'\n#{2,3} ', RECIPE.split(f'id: {step_id}\n', 1)[1], maxsplit=1)[0]
    return re.search(r'^```bash\n(.*?)^```', section, re.M | re.S).group(1)


class RecipeContract(unittest.TestCase):
    def test_included_operations_have_unique_matching_ansible_tasks(self):
        tasks = []
        for path in sorted((ROOT / 'roles').glob('*/tasks/*.yml')):
            tasks += re.findall(r'^\s*- name: (amd\.[^\n]+)$', path.read_text(), re.M)
        task_ids = [name.split(' | ', 1)[0] for name in tasks]
        metadata = re.findall(r'<!-- ops-step\nid: ([^\n]+)\nansible_task: ([^\n]+)\nkind: ([^\n]+)\n-->', RECIPE)
        self.assertEqual(len(task_ids), len(set(task_ids)))
        self.assertEqual(len(metadata), len({entry[0] for entry in metadata}))
        manual_ids = {entry[0] for entry in metadata}
        validation_only = {
            'amd.driver.settings.require',
            'amd.rocm.settings.require',
            'amd.rocm.discovery.stat',
            'amd.rocm.discovery.pending',
            'amd.rocm.packages.inspect',
            'amd.rocm.smoke.settings.require',
            'amd.rocm.smoke.pending',
        }
        self.assertEqual(set(task_ids) - manual_ids, validation_only)
        self.assertLessEqual(manual_ids, set(task_ids))
        self.assertLessEqual({entry[1] for entry in metadata}, set(tasks))

    def test_numbering_matches_role_order_and_groups_command_blocks(self):
        roles = re.findall(r'^\s+- role: (\w+)$', (ROOT / 'site.yml').read_text(), re.M)
        headings = list(re.finditer(r'^## (\d+)\. (.+)$', RECIPE, re.M))
        self.assertEqual([int(item[1]) for item in headings], list(range(1, len(roles) + 1)))
        for index, (heading, role) in enumerate(zip(headings, roles)):
            end = headings[index + 1].start() if index + 1 < len(headings) else len(RECIPE)
            section = RECIPE[heading.end():end]
            self.assertIn(f'(roles/{role}/)', section)
            parts = list(re.finditer(r'^### (\d+)\.(\d+) .+$', section, re.M))
            self.assertTrue(parts)
            self.assertEqual([int(part[1]) for part in parts], [index + 1] * len(parts))
            self.assertEqual([int(part[2]) for part in parts], list(range(1, len(parts) + 1)))
            for part_index, part in enumerate(parts):
                part_end = parts[part_index + 1].start() if part_index + 1 < len(parts) else len(section)
                self.assertLessEqual(section[part.end():part_end].count('```bash\n'), 1)

    def test_manual_shell_blocks_parse(self):
        for index, block in enumerate(re.findall(r'^```bash\n(.*?)^```', RECIPE, re.M | re.S)):
            with self.subTest(block=index):
                result = subprocess.run(['bash', '-n'], input=block, text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_legacy_version_pattern_matches_only_the_reviewed_build(self):
        pattern = re.search(r"ROCM_LEGACY_VERSION_PATTERN='([^']+)'", RECIPE).group(1)
        for version in ('7.2.0.70200-43~24.04', '1.2.1.70200-43~24.04', '4.1.70200.70200-43~24.04'):
            self.assertRegex(version, pattern)
        for version in ('7.1.0-0ubuntu6', '7.2.0.70200-43~26.04', '7.2.0.70200-43~24x04'):
            self.assertNotRegex(version, pattern)

    def test_inline_sources_match_the_deployed_files(self):
        access = re.search(r"<<'PY'\n(.*?)\nPY", step_shell('amd.driver.verify.access'), re.S).group(1)
        self.assertEqual(access + '\n', (ROOT / 'roles/amdgpu_preflight/files/check_access.py').read_text())
        source = re.search(r"<<'CPP'\n(.*?)\nCPP", step_shell('amd.rocm.smoke.source'), re.S).group(1)
        self.assertEqual(source + '\n', (ROOT / 'roles/rocm_smoke_test/files/vector_add.cpp').read_text())

    def test_document_links_and_playbook_commands(self):
        for path in ROOT.glob('*.md'):
            content = path.read_text()
            for link in re.findall(r'\]\(([^)]+)\)', content):
                if not link.startswith(('http://', 'https://', '#')):
                    self.assertTrue((ROOT / link.split('#')[0]).exists(), (path.name, link))
            for line in content.splitlines():
                if line.startswith('ansible-playbook '):
                    self.assertIn('--diff', line)


@unittest.skipUnless(shutil.which('ansible-playbook'), 'ansible-playbook is required')
class TransactionGuards(unittest.TestCase):
    cases = [
        ('clean_install', 'Inst rocm-dev (7.1 Ubuntu:26.04/resolute [amd64])\n', True),
        ('reviewed_cleanup', 'Remv rocm-core [7.2]\nInst rocm-dev (7.1 Ubuntu:26.04/resolute [amd64])\n', True),
        ('already_installed', '0 upgraded, 0 newly installed, 0 to remove\n', True),
        ('unrelated_removal', 'Remv openssh-server [1.0]\n', False),
        ('kernel_change', 'Inst linux-image-7.0 (7.0 Ubuntu:26.04/resolute [amd64])\n', False),
        ('driver_change', 'Inst amdgpu-dkms (1.0 Ubuntu:26.04/resolute [all])\n', False),
        ('installed_package_upgrade', 'Inst libc6 [2.40] (2.43 Ubuntu:26.04/resolute [amd64])\n', False),
        ('vendor_package', 'Inst rocm-dev (7.2 AMD:24.04/noble [amd64])\n', False),
    ]

    def test_manual_and_ansible_accept_the_same_plans(self):
        manual = re.search(r"<<'PLAN'\n(.*?)\nPLAN", step_shell('amd.rocm.transaction.validate'), re.S).group(1)
        with tempfile.TemporaryDirectory(prefix='amd-contract-') as work:
            env = dict(os.environ, ANSIBLE_LOCAL_TEMP=str(Path(work) / 'local'),
                       ANSIBLE_REMOTE_TEMP=str(Path(work) / 'remote'), ANSIBLE_NOCOLOR='1',
                       ROCM_LEGACY_PACKAGES='rocm-core')
            for name, plan, accepted in self.cases:
                with self.subTest(case=name):
                    result = subprocess.run([sys.executable, '-c', manual],
                                            env=dict(env, ROCM_TRANSACTION=plan),
                                            text=True, capture_output=True)
                    self.assertEqual(result.returncode == 0, accepted, result.stderr)
                    play = [{
                        'name': 'Exercise transaction validation only',
                        'hosts': 'localhost', 'gather_facts': False,
                        'vars': {'rocm_transaction': {'rc': 0, 'stdout': plan},
                                 'rocm_legacy_packages': ['rocm-core']},
                        'tasks': [{'ansible.builtin.import_tasks': str(ROOT / 'roles/rocm/tasks/validate_transaction.yml')}],
                    }]
                    path = Path(work) / 'test.yml'
                    path.write_text(json.dumps(play))
                    result = subprocess.run(['ansible-playbook', '-i', 'localhost,',
                                             '--connection', 'local', str(path), '--diff'],
                                            cwd=ROOT, env=env, text=True, capture_output=True)
                    self.assertEqual(result.returncode == 0, accepted, result.stdout + result.stderr)
                    if not accepted:
                        self.assertIn('Review the simulation before continuing.', result.stdout)


if __name__ == '__main__':
    unittest.main()
