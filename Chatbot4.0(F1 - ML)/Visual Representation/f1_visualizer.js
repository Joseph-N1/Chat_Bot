document.getElementById('jsonFileInput').addEventListener('change', async function (event) {
  const file = event.target.files[0];
  if (!file) return;

  const text = await file.text();
  const data = JSON.parse(text);
  visualizeF1Season(data);
});

function visualizeF1Season(seasonData) {
  const races = seasonData;
  const driverProgression = {};
  const constructorPoints = {};
  const heatmapData = [];
  const dnfCounts = {};

  races.forEach((race, roundIndex) => {
    const raceName = race.race_name;
    let section = null;

    race.data.forEach(entry => {
      if (entry.section) {
        section = entry.section;
      } else if (section === "championship_standings") {
        // Build driver progression
        const driver = entry.driver_name;
        if (!driverProgression[driver]) driverProgression[driver] = [];
        driverProgression[driver].push(entry.pts);

        // Constructor aggregation
        const team = entry.team_name;
        if (!constructorPoints[team]) constructorPoints[team] = [];
        constructorPoints[team].push(entry.pts);
      } else if (section === "race_results") {
        const driver = entry.driver;
        const dnf = entry.dnf === "Yes";
        if (dnf) {
          if (!dnfCounts[driver]) dnfCounts[driver] = 0;
          dnfCounts[driver]++;
        }

        heatmapData.push({
          x: roundIndex + 1,
          y: driver,
          z: entry["starting position"] - entry["finish position"] // positional change
        });
      }
    });
  });

  renderDriverChart(driverProgression);
  renderConstructorChart(constructorPoints);
  renderHeatmap(heatmapData);
  renderDNFs(dnfCounts);

  generateCommentary(driverProgression, constructorPoints, heatmapData, dnfCounts);
}

// Chart functions
function renderDriverChart(data) {
  const traces = Object.entries(data).map(([driver, pts]) => ({
    x: pts.map((_, i) => i + 1),
    y: pts,
    mode: 'lines+markers',
    name: driver
  }));

  Plotly.newPlot('driverProgression', traces, { title: 'Driver Points by Round' });
}

function renderConstructorChart(data) {
  const traces = Object.entries(data).map(([team, pts]) => ({
    x: pts.map((_, i) => i + 1),
    y: pts,
    mode: 'lines+markers',
    name: team
  }));

  Plotly.newPlot('constructorTrends', traces, { title: 'Constructor Points by Round' });
}

function renderHeatmap(data) {
  const x = [...new Set(data.map(d => d.x))];
  const y = [...new Set(data.map(d => d.y))];

  const z = y.map(driver =>
    x.map(round => {
      const match = data.find(d => d.x === round && d.y === driver);
      return match ? match.z : 0;
    })
  );

  Plotly.newPlot('qualRaceHeatmap', [{
    x,
    y,
    z,
    type: 'heatmap',
    colorscale: 'RdBu'
  }], { title: 'Qualifying vs Race Positional Gain/Loss' });
}

function renderDNFs(data) {
  const drivers = Object.keys(data);
  const counts = drivers.map(d => data[d]);

  Plotly.newPlot('dnfChart', [{
    x: drivers,
    y: counts,
    type: 'bar'
  }], { title: 'DNFs per Driver' });
}

// Commentary generation placeholder
function generateCommentary(driverData, constructorData, heatmap, dnfs) {
  document.getElementById('driverCommentary').innerText = "[Driver insights will be generated here using GPT]";
  document.getElementById('constructorCommentary').innerText = "[Constructor trends commentary will appear here]";
  document.getElementById('qualCommentary').innerText = "[Heatmap analysis summary here]";
  document.getElementById('dnfCommentary').innerText = "[Driver reliability/DNF summary here]";
}
