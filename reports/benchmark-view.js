// Materialize references for the UI; the published store owns every record.
window.BENCHMARK_STORE = window.BENCHMARK_DATA;
window.BENCHMARK_DATA = (() => {
  const store = window.BENCHMARK_STORE;
  function expand(value) {
    if (Array.isArray(value)) return value.map(expand);
    if (!value || typeof value !== 'object') return value;
    if (value.$merge) return Object.assign({},...value.$merge.map(expand));
    if (value.kind === 'rows' && value.columns) {
      const result = Object.fromEntries(Object.entries(value).filter(([k]) => !['columns','rows','row_attributes','cell_overrides'].includes(k)).map(([k,v])=>[k,expand(v)]));
      result.rows = value.rows.map((id,i) => {
        const record = expand(store.measurements[id]);
        return {attempt_id:record.attempt_id || id, cells:value.columns.map((column,j) => {
          return {value:record[column.source][column.name]};
        })};
      });
      return result;
    }
    if (value.$ref) {
      let result = store[value.$ref][value.id];
      for (const key of value.path || []) result = result[key];
      result = expand(result);
      if (value.split) result = result.split(value.split[0])[value.split[1]];
      if (value.offset !== undefined) result += value.offset;
      if (value.commas) result = Number(result).toLocaleString('en-US');
      if (value.scale !== undefined) result = Number((Number(result) * value.scale).toFixed(2));
      if (value.fixed !== undefined) result = Number(result).toFixed(value.fixed);
      return result;
    }
    return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, expand(item)]));
  }
  return expand(store.catalog);
})();

window.benchmarkModelId = value => window.BENCHMARK_DATA.identities.models[value] || value;
