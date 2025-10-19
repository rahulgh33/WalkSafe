import React, { useState } from "react";

export default function App() {
  const [startLat, setStartLat] = useState("41.7796");
  const [startLon, setStartLon] = useState("-87.6636");
  const [endLat, setEndLat] = useState("41.7681");
  const [endLon, setEndLon] = useState("-87.6435");
  const [lambdaVal, setLambdaVal] = useState("1000");
  const [result, setResult] = useState(null);

  async function getRoute() {
    const qs = new URLSearchParams({
      start_lat: startLat,
      start_lon: startLon,
      end_lat: endLat,
      end_lon: endLon,
      lambda: lambdaVal,
    }).toString();

    const res = await fetch(`/route?${qs}`);
    if (!res.ok) {
      setResult({ error: `status ${res.status}` });
      return;
    }
    const json = await res.json();
    setResult(json);
  }

  function openMap() {
    const qs = new URLSearchParams({
      start_lat: startLat,
      start_lon: startLon,
      end_lat: endLat,
      end_lon: endLon,
    }).toString();
    window.open(`/route_map?${qs}`, "_blank");
  }

  return (
    <div style={{ padding: 24, fontFamily: "system-ui, Arial" }}>
      <h1>WalkSafe (dev)</h1>

      <div style={{ marginBottom: 8 }}>
        <label>Start lat/lon: </label>
        <input value={startLat} onChange={(e)=>setStartLat(e.target.value)} style={{width:100}} />
        <input value={startLon} onChange={(e)=>setStartLon(e.target.value)} style={{width:100, marginLeft:8}} />
      </div>

      <div style={{ marginBottom: 8 }}>
        <label>End lat/lon: </label>
        <input value={endLat} onChange={(e)=>setEndLat(e.target.value)} style={{width:100}} />
        <input value={endLon} onChange={(e)=>setEndLon(e.target.value)} style={{width:100, marginLeft:8}} />
      </div>

      <div style={{ marginBottom: 12 }}>
        <label>Lambda: </label>
        <input value={lambdaVal} onChange={(e)=>setLambdaVal(e.target.value)} style={{width:120}} />
      </div>

      <button onClick={getRoute} style={{ marginRight: 8 }}>Get Route (JSON)</button>
      <button onClick={openMap}>Open Route Map</button>

      <pre style={{ marginTop: 16, background: "#f6f6f6", padding: 12 }}>
        {result ? JSON.stringify(result, null, 2) : "No result yet"}
      </pre>
    </div>
  );
}