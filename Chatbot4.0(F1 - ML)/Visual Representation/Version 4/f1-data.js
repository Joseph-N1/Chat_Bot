const f1Data = {
    // Full JSON structure as shown in the HTML
    "canada": {
  "round": 10,
  "name": "Canadian GP",
  "date": "June 13-15, 2025",
  "track": "Circuit Gilles-Villeneuve, Montréal",
  "trackLength": 4.361,
  "weather": "Dry",
  "safetyCars": 1,
  "winner": {
    "driver": "George Russell",
    "team": "Mercedes"
  },
  "fastestLap": {
    "driver": "George Russell",
    "time": "1:14.119",
    "lap": 63
  },
  "results": {
    "qualifying": [
      {"pos": 1, "driver": "George Russell", "team": "Mercedes", "q1": "1:12.075", "q2": "1:11.570", "q3": "1:10.899"},
      {"pos": 2, "driver": "Max Verstappen", "team": "Red Bull", "q1": "1:12.054", "q2": "1:11.638", "q3": "1:11.059"},
      {"pos": 3, "driver": "Oscar Piastri", "team": "McLaren", "q1": "1:11.939", "q2": "1:11.715", "q3": "1:11.120"},
      {"pos": 4, "driver": "Kimi Antonelli", "team": "Mercedes", "q1": "1:12.279", "q2": "1:11.974", "q3": "1:11.391"},
      {"pos": 5, "driver": "Lewis Hamilton", "team": "Ferrari", "q1": "1:11.952", "q2": "1:11.885", "q3": "1:11.526"},
      {"pos": 6, "driver": "Fernando Alonso", "team": "Aston Martin", "q1": "1:12.073", "q2": "1:11.805", "q3": "1:11.586"},
      {"pos": 7, "driver": "Lando Norris", "team": "McLaren", "q1": "1:11.826", "q2": "1:11.599", "q3": "1:11.625"},
      {"pos": 8, "driver": "Charles Leclerc", "team": "Ferrari", "q1": "1:12.038", "q2": "1:11.626", "q3": "1:11.682"},
      {"pos": 9, "driver": "Isack Hadjar", "team": "RB", "q1": "1:12.211", "q2": "1:12.003", "q3": "1:11.867"},
      {"pos": 10, "driver": "Alexander Albon", "team": "Williams", "q1": "1:12.090", "q2": "1:11.892", "q3": "1:11.907"},
      {"pos": 11, "driver": "Yuki Tsunoda", "team": "Red Bull", "q1": "1:12.334", "q2": "1:12.102", "q3": null},
      {"pos": 12, "driver": "Franco Colapinto", "team": "Alpine", "q1": "1:12.234", "q2": "1:12.142", "q3": null},
      {"pos": 13, "driver": "Nico Hulkenberg", "team": "Sauber", "q1": "1:12.323", "q2": "1:12.183", "q3": null},
      {"pos": 14, "driver": "Oliver Bearman", "team": "Haas", "q1": "1:12.306", "q2": "1:12.340", "q3": null},
      {"pos": 15, "driver": "Esteban Ocon", "team": "Haas", "q1": "1:12.378", "q2": "1:12.634", "q3": null},
      {"pos": 16, "driver": "Gabriel Bortoleto", "team": "Sauber", "q1": "1:12.385", "q2": null, "q3": null},
      {"pos": 17, "driver": "Carlos Sainz", "team": "Williams", "q1": "1:12.398", "q2": null, "q3": null},
      {"pos": 18, "driver": "Lance Stroll", "team": "Aston Martin", "q1": "1:12.517", "q2": null, "q3": null},
      {"pos": 19, "driver": "Liam Lawson", "team": "RB", "q1": "1:12.525", "q2": null, "q3": null},
      {"pos": 20, "driver": "Pierre Gasly", "team": "Alpine", "q1": "1:12.667", "q2": null, "q3": null}
    ],
    "race": [
      {"pos": 1, "driver": "George Russell", "team": "Mercedes", "start": 1, "laps": 70, "points": 25},
      {"pos": 2, "driver": "Max Verstappen", "team": "Red Bull", "start": 2, "laps": 70, "points": 18},
      {"pos": 3, "driver": "Kimi Antonelli", "team": "Mercedes", "start": 4, "laps": 70, "points": 15},
      {"pos": 4, "driver": "Oscar Piastri", "team": "McLaren", "start": 3, "laps": 70, "points": 12},
      {"pos": 5, "driver": "Charles Leclerc", "team": "Ferrari", "start": 8, "laps": 70, "points": 10},
      {"pos": 6, "driver": "Lewis Hamilton", "team": "Ferrari", "start": 5, "laps": 70, "points": 8},
      {"pos": 7, "driver": "Fernando Alonso", "team": "Aston Martin", "start": 6, "laps": 70, "points": 6},
      {"pos": 8, "driver": "Nico Hulkenberg", "team": "Sauber", "start": 11, "laps": 70, "points": 4},
      {"pos": 9, "driver": "Esteban Ocon", "team": "Haas", "start": 14, "laps": 69, "points": 2},
      {"pos": 10, "driver": "Carlos Sainz", "team": "Williams", "start": 16, "laps": 69, "points": 1},
      {"pos": 11, "driver": "Oliver Bearman", "team": "Haas", "start": 13, "laps": 69, "points": 0},
      {"pos": 12, "driver": "Yuki Tsunoda", "team": "Red Bull", "start": 18, "laps": 69, "points": 0},
      {"pos": 13, "driver": "Franco Colapinto", "team": "Alpine", "start": 10, "laps": 69, "points": 0},
      {"pos": 14, "driver": "Gabriel Bortoleto", "team": "Sauber", "start": 15, "laps": 69, "points": 0},
      {"pos": 15, "driver": "Pierre Gasly", "team": "Alpine", "start": 20, "laps": 69, "points": 0},
      {"pos": 16, "driver": "Isack Hadjar", "team": "RB", "start": 12, "laps": 69, "points": 0},
      {"pos": 17, "driver": "Lance Stroll", "team": "Aston Martin", "start": 17, "laps": 69, "points": 0},
      {"pos": 18, "driver": "Lando Norris", "team": "McLaren", "start": 7, "laps": 66, "points": 0},
      {"pos": 19, "driver": "Liam Lawson", "team": "RB", "start": 19, "laps": 53, "points": 0},
      {"pos": 20, "driver": "Alexander Albon", "team": "Williams", "start": 9, "laps": 46, "points": 0}
    ]
  },
  "highlights": [
    "George Russell wins dramatic Canadian GP after late-race battle with Verstappen",
    "Mercedes double podium with Antonelli securing 3rd place",
    "Lando Norris DNF after collision and 5-second penalty",
    "Multiple pit lane start penalties applied pre-race",
    "Safety car deployed after Albon retirement on lap 46"
  ],
  "practice": {
    "p1": [
      {"driver": "Max Verstappen", "team": "Red Bull", "time": "1:13.193", "laps": 28},
      {"driver": "Alexander Albon", "team": "Williams", "time": "1:13.232", "laps": 28},
      {"driver": "Carlos Sainz", "team": "Williams", "time": "1:13.275", "laps": 31},
      {"driver": "George Russell", "team": "Mercedes", "time": "1:13.535", "laps": 29},
      {"driver": "Lewis Hamilton", "team": "Ferrari", "time": "1:13.620", "laps": 30},
      {"driver": "Isack Hadjar", "team": "RB", "time": "1:13.631", "laps": 31},
      {"driver": "Lando Norris", "team": "McLaren", "time": "1:13.651", "laps": 30},
      {"driver": "Liam Lawson", "team": "RB", "time": "1:13.737", "laps": 30},
      {"driver": "Pierre Gasly", "team": "Alpine", "time": "1:13.817", "laps": 29},
      {"driver": "Charles Leclerc", "team": "Ferrari", "time": "1:13.885", "laps": 9},
      {"driver": "Yuki Tsunoda", "team": "Red Bull", "time": "1:13.927", "laps": 27},
      {"driver": "Fernando Alonso", "team": "Aston Martin", "time": "1:13.972", "laps": 25},
      {"driver": "Kimi Antonelli", "team": "Mercedes", "time": "1:14.002", "laps": 30},
      {"driver": "Oscar Piastri", "team": "McLaren", "time": "1:14.198", "laps": 28},
      {"driver": "Lance Stroll", "team": "Aston Martin", "time": "1:14.203", "laps": 25},
      {"driver": "Gabriel Bortoleto", "team": "Sauber", "time": "1:14.324", "laps": 30},
      {"driver": "Oliver Bearman", "team": "Haas", "time": "1:14.520", "laps": 30},
      {"driver": "Esteban Ocon", "team": "Haas", "time": "1:14.605", "laps": 23},
      {"driver": "Franco Colapinto", "team": "Alpine", "time": "1:14.645", "laps": 29},
      {"driver": "Nico Hulkenberg", "team": "Sauber", "time": "1:14.821", "laps": 28}
    ],
    "p2": [
      {"driver": "George Russell", "team": "Mercedes", "time": "1:12.123", "laps": 33},
      {"driver": "Lando Norris", "team": "McLaren", "time": "1:12.151", "laps": 32},
      {"driver": "Kimi Antonelli", "team": "Mercedes", "time": "1:12.411", "laps": 33},
      {"driver": "Alexander Albon", "team": "Williams", "time": "1:12.445", "laps": 36},
      {"driver": "Fernando Alonso", "team": "Aston Martin", "time": "1:12.458", "laps": 31},
      {"driver": "Oscar Piastri", "team": "McLaren", "time": "1:12.562", "laps": 32},
      {"driver": "Carlos Sainz", "team": "Williams", "time": "1:12.631", "laps": 37},
      {"driver": "Lewis Hamilton", "team": "Ferrari", "time": "1:12.653", "laps": 34},
      {"driver": "Max Verstappen", "team": "Red Bull", "time": "1:12.666", "laps": 31},
      {"driver": "Liam Lawson", "team": "RB", "time": "1:12.751", "laps": 30},
      {"driver": "Isack Hadjar", "team": "RB", "time": "1:12.799", "laps": 31},
      {"driver": "Pierre Gasly", "team": "Alpine", "time": "1:12.874", "laps": 34},
      {"driver": "Gabriel Bortoleto", "team": "Sauber", "time": "1:12.896", "laps": 32},
      {"driver": "Nico Hulkenberg", "team": "Sauber", "time": "1:12.914", "laps": 33},
      {"driver": "Yuki Tsunoda", "team": "Red Bull", "time": "1:12.939", "laps": 35},
      {"driver": "Oliver Bearman", "team": "Haas", "time": "1:13.080", "laps": 36},
      {"driver": "Esteban Ocon", "team": "Haas", "time": "1:13.175", "laps": 33},
      {"driver": "Franco Colapinto", "team": "Alpine", "time": "1:13.898", "laps": 33},
      {"driver": "Lance Stroll", "team": "Aston Martin", "time": "No time", "laps": 2}
    ],
    "p3": [
      {"driver": "Lando Norris", "team": "McLaren", "time": "1:11.799", "laps": 24},
      {"driver": "Charles Leclerc", "team": "Ferrari", "time": "1:11.877", "laps": 29},
      {"driver": "George Russell", "team": "Mercedes", "time": "1:11.950", "laps": 20},
      {"driver": "Lewis Hamilton", "team": "Ferrari", "time": "1:12.050", "laps": 26},
      {"driver": "Max Verstappen", "team": "Red Bull", "time": "1:12.072", "laps": 20},
      {"driver": "Fernando Alonso", "team": "Aston Martin", "time": "1:12.247", "laps": 21},
      {"driver": "Kimi Antonelli", "team": "Mercedes", "time": "1:12.348", "laps": 21},
      {"driver": "Oscar Piastri", "team": "McLaren", "time": "1:12.519", "laps": 18},
      {"driver": "Carlos Sainz", "team": "Williams", "time": "1:12.519", "laps": 22},
      {"driver": "Alexander Albon", "team": "Williams", "time": "1:12.573", "laps": 22},
      {"driver": "Isack Hadjar", "team": "RB", "time": "1:12.651", "laps": 22},
      {"driver": "Pierre Gasly", "team": "Alpine", "time": "1:12.684", "laps": 27},
      {"driver": "Liam Lawson", "team": "RB", "time": "1:12.791", "laps": 27},
      {"driver": "Lance Stroll", "team": "Aston Martin", "time": "1:12.794", "laps": 28},
      {"driver": "Oliver Bearman", "team": "Haas", "time": "1:12.825", "laps": 27},
      {"driver": "Esteban Ocon", "team": "Haas", "time": "1:12.827", "laps": 22},
      {"driver": "Franco Colapinto", "team": "Alpine", "time": "1:13.060", "laps": 27},
      {"driver": "Nico Hulkenberg", "team": "Sauber", "time": "1:13.072", "laps": 19},
      {"driver": "Gabriel Bortoleto", "team": "Sauber", "time": "1:13.172", "laps": 22},
      {"driver": "Yuki Tsunoda", "team": "Red Bull", "time": "1:13.573", "laps": 14}
    ]
  },
  "pitStops": [
    {
      "driver": "Max Verstappen",
      "stops": 2,
      "totalTime": 46.725,
      "stopsDetail": [
        {"lap": 12, "time": 23.604},
        {"lap": 37, "time": 23.121}
      ]
    },
    {
      "driver": "George Russell",
      "stops": 2,
      "totalTime": 47.493,
      "stopsDetail": [
        {"lap": 13, "time": 23.231},
        {"lap": 42, "time": 24.262}
      ]
    },
    {
      "driver": "Kimi Antonelli",
      "stops": 2,
      "totalTime": 46.736,
      "stopsDetail": [
        {"lap": 14, "time": 23.320},
        {"lap": 38, "time": 23.416}
      ]
    },
    {
      "driver": "Oscar Piastri",
      "stops": 3,
      "totalTime": 72.804,
      "stopsDetail": [
        {"lap": 16, "time": 23.245},
        {"lap": 45, "time": 23.174},
        {"lap": 67, "time": 26.385}
      ]
    },
    {
      "driver": "Lewis Hamilton",
      "stops": 2,
      "totalTime": 46.942,
      "stopsDetail": [
        {"lap": 15, "time": 23.604},
        {"lap": 45, "time": 23.338}
      ]
    },
    {
      "driver": "Fernando Alonso",
      "stops": 2,
      "totalTime": 47.430,
      "stopsDetail": [
        {"lap": 15, "time": 24.021},
        {"lap": 50, "time": 23.409}
      ]
    },
    {
      "driver": "Charles Leclerc",
      "stops": 2,
      "totalTime": 47.219,
      "stopsDetail": [
        {"lap": 28, "time": 23.360},
        {"lap": 53, "time": 23.859}
      ]
    },
    {
      "driver": "Lando Norris",
      "stops": 2,
      "totalTime": 46.694,
      "stopsDetail": [
        {"lap": 29, "time": 23.223},
        {"lap": 47, "time": 23.471}
      ]
    },
    {
      "driver": "Alexander Albon",
      "stops": 1,
      "totalTime": 23.898,
      "stopsDetail": [
        {"lap": 23, "time": 23.898}
      ]
    },
    {
      "driver": "Lance Stroll",
      "stops": 3,
      "totalTime": 82.256,
      "stopsDetail": [
        {"lap": 24, "time": 23.742},
        {"lap": 51, "time": 34.742},
        {"lap": 66, "time": 23.772}
      ]
    },
    {
      "driver": "Nico Hulkenberg",
      "stops": 1,
      "totalTime": 23.476,
      "stopsDetail": [
        {"lap": 19, "time": 23.476}
      ]
    },
    {
      "driver": "Esteban Ocon",
      "stops": 1,
      "totalTime": 24.452,
      "stopsDetail": [
        {"lap": 57, "time": 24.452}
      ]
    },
    {
      "driver": "Carlos Sainz",
      "stops": 1,
      "totalTime": 23.269,
      "stopsDetail": [
        {"lap": 57, "time": 23.269}
      ]
    },
    {
      "driver": "Oliver Bearman",
      "stops": 2,
      "totalTime": 51.547,
      "stopsDetail": [
        {"lap": 18, "time": 23.562},
        {"lap": 66, "time": 27.985}
      ]
    },
    {
      "driver": "Yuki Tsunoda",
      "stops": 1,
      "totalTime": 25.178,
      "stopsDetail": [
        {"lap": 56, "time": 25.178}
      ]
    },
    {
      "driver": "Franco Colapinto",
      "stops": 1,
      "totalTime": 24.735,
      "stopsDetail": [
        {"lap": 14, "time": 24.735}
      ]
    },
    {
      "driver": "Gabriel Bortoleto",
      "stops": 1,
      "totalTime": 23.715,
      "stopsDetail": [
        {"lap": 49, "time": 23.715}
      ]
    },
    {
      "driver": "Pierre Gasly",
      "stops": 1,
      "totalTime": 23.987,
      "stopsDetail": [
        {"lap": 53, "time": 23.987}
      ]
    },
    {
      "driver": "Isack Hadjar",
      "stops": 2,
      "totalTime": 47.823,
      "stopsDetail": [
        {"lap": 13, "time": 24.130},
        {"lap": 66, "time": 23.693}
      ]
    },
    {
      "driver": "Liam Lawson",
      "stops": 1,
      "totalTime": 23.280,
      "stopsDetail": [
        {"lap": 38, "time": 23.280}
      ]
    }
  ],
  "fastestLaps": [
    {"driver": "George Russell", "lapTime": "1:14.119", "lap": 63, "avgSpeed": "211.816 km/h"},
    {"driver": "Lando Norris", "lapTime": "1:14.229", "lap": 65, "avgSpeed": "211.502 km/h"},
    {"driver": "Oscar Piastri", "lapTime": "1:14.255", "lap": 64, "avgSpeed": "211.428 km/h"},
    {"driver": "Charles Leclerc", "lapTime": "1:14.261", "lap": 57, "avgSpeed": "211.411 km/h"},
    {"driver": "Max Verstappen", "lapTime": "1:14.287", "lap": 62, "avgSpeed": "211.337 km/h"},
    {"driver": "Carlos Sainz", "lapTime": "1:14.389", "lap": 60, "avgSpeed": "211.047 km/h"},
    {"driver": "Kimi Antonelli", "lapTime": "1:14.455", "lap": 60, "avgSpeed": "210.860 km/h"},
    {"driver": "Esteban Ocon", "lapTime": "1:14.593", "lap": 62, "avgSpeed": "210.470 km/h"},
    {"driver": "Lewis Hamilton", "lapTime": "1:14.805", "lap": 64, "avgSpeed": "209.873 km/h"},
    {"driver": "Lance Stroll", "lapTime": "1:14.902", "lap": 58, "avgSpeed": "209.601 km/h"},
    {"driver": "Pierre Gasly", "lapTime": "1:14.993", "lap": 64, "avgSpeed": "209.347 km/h"},
    {"driver": "Fernando Alonso", "lapTime": "1:15.024", "lap": 58, "avgSpeed": "209.261 km/h"},
    {"driver": "Yuki Tsunoda", "lapTime": "1:15.358", "lap": 60, "avgSpeed": "208.333 km/h"},
    {"driver": "Nico Hulkenberg", "lapTime": "1:15.372", "lap": 65, "avgSpeed": "208.294 km/h"},
    {"driver": "Oliver Bearman", "lapTime": "1:15.397", "lap": 63, "avgSpeed": "208.225 km/h"},
    {"driver": "Gabriel Bortoleto", "lapTime": "1:15.414", "lap": 57, "avgSpeed": "208.178 km/h"},
    {"driver": "Franco Colapinto", "lapTime": "1:16.076", "lap": 53, "avgSpeed": "206.367 km/h"},
    {"driver": "Alexander Albon", "lapTime": "1:16.197", "lap": 31, "avgSpeed": "206.039 km/h"},
    {"driver": "Isack Hadjar", "lapTime": "1:16.292", "lap": 51, "avgSpeed": "205.783 km/h"},
    {"driver": "Liam Lawson", "lapTime": "1:16.320", "lap": 53, "avgSpeed": "205.707 km/h"}
  ]
},
"spain": {
  "round": 9,
  "name": "Spanish GP",
  "date": "May 30 - June 1, 2025",
  "track": "Circuit de Barcelona-Catalunya",
  "trackLength": 4.675,
  "weather": "Dry",
  "safetyCars": 0,
  "winner": {
    "driver": "Oscar Piastri",
    "team": "McLaren"
  },
  "fastestLap": {
    "driver": "Oscar Piastri",
    "time": "1:15.743",
    "lap": 61
  },
  "results": {
    "qualifying": [
      {"pos": 1, "driver": "Oscar Piastri", "team": "McLaren", "q1": "1:12.551", "q2": "1:11.998", "q3": "1:11.546"},
      {"pos": 2, "driver": "Lando Norris", "team": "McLaren", "q1": "1:12.799", "q2": "1:12.056", "q3": "1:11.755"},
      {"pos": 3, "driver": "Max Verstappen", "team": "Red Bull", "q1": "1:12.798", "q2": "1:12.358", "q3": "1:11.848"},
      {"pos": 4, "driver": "George Russell", "team": "Mercedes", "q1": "1:12.806", "q2": "1:12.407", "q3": "1:11.848"},
      {"pos": 5, "driver": "Lewis Hamilton", "team": "Ferrari", "q1": "1:13.058", "q2": "1:12.447", "q3": "1:12.045"},
      {"pos": 6, "driver": "Kimi Antonelli", "team": "Mercedes", "q1": "1:12.815", "q2": "1:12.585", "q3": "1:12.111"},
      {"pos": 7, "driver": "Charles Leclerc", "team": "Ferrari", "q1": "1:13.014", "q2": "1:12.495", "q3": "1:12.131"},
      {"pos": 8, "driver": "Pierre Gasly", "team": "Alpine", "q1": "1:13.081", "q2": "1:12.611", "q3": "1:12.199"},
      {"pos": 9, "driver": "Isack Hadjar", "team": "RB", "q1": "1:13.139", "q2": "1:12.461", "q3": "1:12.252"},
      {"pos": 10, "driver": "Fernando Alonso", "team": "Aston Martin", "q1": "1:13.102", "q2": "1:12.523", "q3": "1:12.284"},
      {"pos": 11, "driver": "Alexander Albon", "team": "Williams", "q1": "1:13.044", "q2": "1:12.641", "q3": null},
      {"pos": 12, "driver": "Gabriel Bortoleto", "team": "Sauber", "q1": "1:13.045", "q2": "1:12.756", "q3": null},
      {"pos": 13, "driver": "Liam Lawson", "team": "RB", "q1": "1:13.039", "q2": "1:12.763", "q3": null},
      {"pos": 14, "driver": "Lance Stroll", "team": "Aston Martin", "q1": "1:13.038", "q2": "1:13.058", "q3": null},
      {"pos": 15, "driver": "Oliver Bearman", "team": "Haas", "q1": "1:13.074", "q2": "1:13.315", "q3": null},
      {"pos": 16, "driver": "Nico Hulkenberg", "team": "Sauber", "q1": "1:13.190", "q2": null, "q3": null},
      {"pos": 17, "driver": "Esteban Ocon", "team": "Haas", "q1": "1:13.201", "q2": null, "q3": null},
      {"pos": 18, "driver": "Carlos Sainz", "team": "Williams", "q1": "1:13.203", "q2": null, "q3": null},
      {"pos": 19, "driver": "Franco Colapinto", "team": "Alpine", "q1": "1:13.334", "q2": null, "q3": null},
      {"pos": 20, "driver": "Yuki Tsunoda", "team": "Red Bull", "q1": "1:13.385", "q2": null, "q3": null}
    ],
    "race": [
      {"pos": 1, "driver": "Oscar Piastri", "team": "McLaren", "start": 1, "laps": 66, "points": 25},
      {"pos": 2, "driver": "Lando Norris", "team": "McLaren", "start": 2, "laps": 66, "points": 18},
      {"pos": 3, "driver": "Max Verstappen", "team": "Red Bull", "start": 3, "laps": 66, "points": 15},
      {"pos": 4, "driver": "George Russell", "team": "Mercedes", "start": 4, "laps": 66, "points": 12},
      {"pos": 5, "driver": "Charles Leclerc", "team": "Ferrari", "start": 7, "laps": 66, "points": 10},
      {"pos": 6, "driver": "Nico Hulkenberg", "team": "Sauber", "start": 15, "laps": 66, "points": 8},
      {"pos": 7, "driver": "Lewis Hamilton", "team": "Ferrari", "start": 5, "laps": 66, "points": 6},
      {"pos": 8, "driver": "Isack Hadjar", "team": "RB", "start": 9, "laps": 66, "points": 4},
      {"pos": 9, "driver": "Pierre Gasly", "team": "Alpine", "start": 8, "laps": 66, "points": 2},
      {"pos": 10, "driver": "Yuki Tsunoda", "team": "Red Bull", "start": 19, "laps": 66, "points": 1},
      {"pos": 11, "driver": "Fernando Alonso", "team": "Aston Martin", "start": 10, "laps": 66, "points": 0},
      {"pos": 12, "driver": "Kimi Antonelli", "team": "Mercedes", "start": 6, "laps": 66, "points": 0},
      {"pos": 13, "driver": "Gabriel Bortoleto", "team": "Sauber", "start": 12, "laps": 66, "points": 0},
      {"pos": 14, "driver": "Franco Colapinto", "team": "Alpine", "start": 18, "laps": 66, "points": 0},
      {"pos": 15, "driver": "Oliver Bearman", "team": "Haas", "start": 14, "laps": 66, "points": 0},
      {"pos": 16, "driver": "Esteban Ocon", "team": "Haas", "start": 16, "laps": 66, "points": 0},
      {"pos": 17, "driver": "Liam Lawson", "team": "RB", "start": 13, "laps": 66, "points": 0},
      {"pos": 18, "driver": "Carlos Sainz", "team": "Williams", "start": 17, "laps": 66, "points": 0},
      {"pos": 19, "driver": "Alexander Albon", "team": "Williams", "start": 11, "laps": 66, "points": 0}
    ]
  },
  "highlights": [
    "McLaren 1-2 with Piastri securing his first career victory",
    "Strategic masterclass from McLaren with perfect tire management",
    "Hulkenberg's impressive P6 for Sauber after starting 15th",
    "Tsunoda recovers from pit lane start to score point",
    "No safety car interventions in clean race"
  ],
  "practice": {
    "p1": [
      {"driver": "Lando Norris", "team": "McLaren", "time": "1:13.718", "laps": 29},
      {"driver": "Max Verstappen", "team": "Red Bull", "time": "1:14.085", "laps": 18},
      {"driver": "Lewis Hamilton", "team": "Ferrari", "time": "1:14.096", "laps": 29},
      {"driver": "Charles Leclerc", "team": "Ferrari", "time": "1:14.238", "laps": 31},
      {"driver": "Oscar Piastri", "team": "McLaren", "time": "1:14.294", "laps": 28},
      {"driver": "Liam Lawson", "team": "RB", "time": "1:14.339", "laps": 28},
      {"driver": "Oliver Bearman", "team": "Haas", "time": "1:14.597", "laps": 26},
      {"driver": "Isack Hadjar", "team": "RB", "time": "1:14.605", "laps": 26},
      {"driver": "Yuki Tsunoda", "team": "Red Bull", "time": "1:14.643", "laps": 27},
      {"driver": "Pierre Gasly", "team": "Alpine", "time": "1:14.746", "laps": 28},
      {"driver": "George Russell", "team": "Mercedes", "time": "1:14.751", "laps": 32},
      {"driver": "Lance Stroll", "team": "Aston Martin", "time": "1:14.786", "laps": 24},
      {"driver": "Fernando Alonso", "team": "Aston Martin", "time": "1:14.798", "laps": 20},
      {"driver": "Nico Hulkenberg", "team": "Sauber", "time": "1:14.865", "laps": 21},
      {"driver": "Carlos Sainz", "team": "Williams", "time": "1:14.935", "laps": 26},
      {"driver": "Gabriel Bortoleto", "team": "Sauber", "time": "1:15.155", "laps": 23},
      {"driver": "Ryo Hirakawa", "team": "Haas", "time": "1:15.298", "laps": 23},
      {"driver": "Kimi Antonelli", "team": "Mercedes", "time": "1:15.369", "laps": 31},
      {"driver": "Victor Martins", "team": "Williams", "time": "1:15.522", "laps": 26},
      {"driver": "Franco Colapinto", "team": "Alpine", "time": "1:15.530", "laps": 19}
    ],
    "p2": [
      {"driver": "Oscar Piastri", "team": "McLaren", "time": "1:12.760", "laps": 28},
      {"driver": "George Russell", "team": "Mercedes", "time": "1:13.046", "laps": 32},
      {"driver": "Max Verstappen", "team": "Red Bull", "time": "1:13.070", "laps": 30},
      {"driver": "Lando Norris", "team": "McLaren", "time": "1:13.070", "laps": 31},
      {"driver": "Charles Leclerc", "team": "Ferrari", "time": "1:13.260", "laps": 33},
      {"driver": "Kimi Antonelli", "team": "Mercedes", "time": "1:13.298", "laps": 31},
      {"driver": "Fernando Alonso", "team": "Aston Martin", "time": "1:13.301", "laps": 28},
      {"driver": "Pierre Gasly", "team": "Alpine", "time": "1:13.385", "laps": 30},
      {"driver": "Isack Hadjar", "team": "RB", "time": "1:13.400", "laps": 29},
      {"driver": "Liam Lawson", "team": "RB", "time": "1:13.494", "laps": 29},
      {"driver": "Lewis Hamilton", "team": "Ferrari", "time": "1:13.533", "laps": 29},
      {"driver": "Nico Hulkenberg", "team": "Sauber", "time": "1:13.592", "laps": 30},
      {"driver": "Yuki Tsunoda", "team": "Red Bull", "time": "1:13.683", "laps": 31},
      {"driver": "Carlos Sainz", "team": "Williams", "time": "1:13.721", "laps": 34},
      {"driver": "Alexander Albon", "team": "Williams", "time": "1:13.839", "laps": 32},
      {"driver": "Lance Stroll", "team": "Aston Martin", "time": "1:13.839", "laps": 17},
      {"driver": "Gabriel Bortoleto", "team": "Sauber", "time": "1:13.959", "laps": 27},
      {"driver": "Esteban Ocon", "team": "Haas", "time": "1:14.005", "laps": 30},
      {"driver": "Oliver Bearman", "team": "Haas", "time": "1:14.126", "laps": 20},
      {"driver": "Franco Colapinto", "team": "Alpine", "time": "1:14.303", "laps": 31}
    ],
    "p3": [
      {"driver": "Oscar Piastri", "team": "McLaren", "time": "1:12.387", "laps": 14},
      {"driver": "Lando Norris", "team": "McLaren", "time": "1:12.913", "laps": 18},
      {"driver": "Charles Leclerc", "team": "Ferrari", "time": "1:13.130", "laps": 17},
      {"driver": "George Russell", "team": "Mercedes", "time": "1:13.139", "laps": 18},
      {"driver": "Max Verstappen", "team": "Red Bull", "time": "1:13.375", "laps": 14},
      {"driver": "Isack Hadjar", "team": "RB", "time": "1:13.382", "laps": 17},
      {"driver": "Kimi Antonelli", "team": "Mercedes", "time": "1:13.405", "laps": 12},
      {"driver": "Fernando Alonso", "team": "Aston Martin", "time": "1:13.414", "laps": 17},
      {"driver": "Lewis Hamilton", "team": "Ferrari", "time": "1:13.527", "laps": 17},
      {"driver": "Liam Lawson", "team": "RB", "time": "1:13.637", "laps": 18},
      {"driver": "Gabriel Bortoleto", "team": "Sauber", "time": "1:13.722", "laps": 19},
      {"driver": "Nico Hulkenberg", "team": "Sauber", "time": "1:13.733", "laps": 18},
      {"driver": "Carlos Sainz", "team": "Williams", "time": "1:13.758", "laps": 16},
      {"driver": "Yuki Tsunoda", "team": "Red Bull", "time": "1:13.892", "laps": 13},
      {"driver": "Lance Stroll", "team": "Aston Martin", "time": "1:13.904", "laps": 20},
      {"driver": "Pierre Gasly", "team": "Alpine", "time": "1:13.954", "laps": 20},
      {"driver": "Franco Colapinto", "team": "Alpine", "time": "1:14.085", "laps": 23},
      {"driver": "Esteban Ocon", "team": "Haas", "time": "1:14.138", "laps": 14},
      {"driver": "Alexander Albon", "team": "Williams", "time": "1:14.289", "laps": 5},
      {"driver": "Oliver Bearman", "team": "Haas", "time": "1:14.460", "laps": 12}
    ]
  },
  "pitStops": [
    {
      "driver": "Max Verstappen",
      "stops": 3,
      "totalTime": 65.604,
      "stopsDetail": [
        {"lap": 13, "time": 21.869},
        {"lap": 29, "time": 21.933},
        {"lap": 47, "time": 21.802}
      ]
    },
    {
      "driver": "Oscar Piastri",
      "stops": 3,
      "totalTime": 66.444,
      "stopsDetail": [
        {"lap": 22, "time": 21.858},
        {"lap": 49, "time": 21.838},
        {"lap": 55, "time": 22.748}
      ]
    },
    {
      "driver": "Lando Norris",
      "stops": 3,
      "totalTime": 67.879,
      "stopsDetail": [
        {"lap": 21, "time": 22.454},
        {"lap": 48, "time": 21.863},
        {"lap": 55, "time": 23.562}
      ]
    },
    {
      "driver": "George Russell",
      "stops": 3,
      "totalTime": 66.658,
      "stopsDetail": [
        {"lap": 20, "time": 21.739},
        {"lap": 41, "time": 21.752},
        {"lap": 55, "time": 23.167}
      ]
    },
    {
      "driver": "Charles Leclerc",
      "stops": 3,
      "totalTime": 66.139,
      "stopsDetail": [
        {"lap": 17, "time": 21.863},
        {"lap": 40, "time": 21.893},
        {"lap": 55, "time": 22.383}
      ]
    },
    {
      "driver": "Nico Hulkenberg",
      "stops": 2,
      "totalTime": 44.082,
      "stopsDetail": [
        {"lap": 9, "time": 22.233},
        {"lap": 45, "time": 21.849}
      ]
    },
    {
      "driver": "Lewis Hamilton",
      "stops": 3,
      "totalTime": 69.203,
      "stopsDetail": [
        {"lap": 16, "time": 21.957},
        {"lap": 46, "time": 24.416},
        {"lap": 55, "time": 22.830}
      ]
    },
    {
      "driver": "Isack Hadjar",
      "stops": 3,
      "totalTime": 65.629,
      "stopsDetail": [
        {"lap": 19, "time": 21.769},
        {"lap": 48, "time": 22.019},
        {"lap": 55, "time": 21.841}
      ]
    },
    {
      "driver": "Pierre Gasly",
      "stops": 3,
      "totalTime": 67.048,
      "stopsDetail": [
        {"lap": 10, "time": 22.224},
        {"lap": 31, "time": 22.006},
        {"lap": 55, "time": 22.818}
      ]
    },
    {
      "driver": "Yuki Tsunoda",
      "stops": 4,
      "totalTime": 88.029,
      "stopsDetail": [
        {"lap": 8, "time": 21.868},
        {"lap": 24, "time": 22.056},
        {"lap": 44, "time": 21.822},
        {"lap": 54, "time": 22.283}
      ]
    },
    {
      "driver": "Fernando Alonso",
      "stops": 3,
      "totalTime": 67.240,
      "stopsDetail": [
        {"lap": 15, "time": 22.782},
        {"lap": 42, "time": 22.242},
        {"lap": 54, "time": 22.216}
      ]
    },
    {
      "driver": "Kimi Antonelli",
      "stops": 2,
      "totalTime": 45.691,
      "stopsDetail": [
        {"lap": 21, "time": 22.234},
        {"lap": 49, "time": 23.457}
      ]
    },
    {
      "driver": "Gabriel Bortoleto",
      "stops": 2,
      "totalTime": 44.994,
      "stopsDetail": [
        {"lap": 19, "time": 22.041},
        {"lap": 49, "time": 22.953}
      ]
    },
    {
      "driver": "Franco Colapinto",
      "stops": 3,
      "totalTime": 66.484,
      "stopsDetail": [
        {"lap": 14, "time": 22.491},
        {"lap": 39, "time": 22.119},
        {"lap": 54, "time": 21.874}
      ]
    },
    {
      "driver": "Oliver Bearman",
      "stops": 3,
      "totalTime": 67.044,
      "stopsDetail": [
        {"lap": 8, "time": 22.009},
        {"lap": 35, "time": 22.270},
        {"lap": 54, "time": 22.765}
      ]
    },
    {
      "driver": "Esteban Ocon",
      "stops": 2,
      "totalTime": 45.022,
      "stopsDetail": [
        {"lap": 20, "time": 22.554},
        {"lap": 43, "time": 22.468}
      ]
    },
    {
      "driver": "Liam Lawson",
      "stops": 2,
      "totalTime": 44.166,
      "stopsDetail": [
        {"lap": 18, "time": 22.039},
        {"lap": 44, "time": 22.127}
      ]
    },
    {
      "driver": "Carlos Sainz",
      "stops": 3,
      "totalTime": 74.877,
      "stopsDetail": [
        {"lap": 9, "time": 30.547},
        {"lap": 34, "time": 22.346},
        {"lap": 55, "time": 21.984}
      ]
    },
    {
      "driver": "Alexander Albon",
      "stops": 2,
      "totalTime": 68.497,
      "stopsDetail": [
        {"lap": 6, "time": 30.823},
        {"lap": 26, "time": 37.674}
      ]
    }
  ],
  "fastestLaps": [
    {"driver": "Oscar Piastri", "lapTime": "1:15.743", "lap": 61, "avgSpeed": "221.343 km/h"},
    {"driver": "Lando Norris", "lapTime": "1:16.187", "lap": 61, "avgSpeed": "220.053 km/h"},
    {"driver": "Max Verstappen", "lapTime": "1:17.019", "lap": 62, "avgSpeed": "217.676 km/h"},
    {"driver": "George Russell", "lapTime": "1:17.244", "lap": 62, "avgSpeed": "217.042 km/h"},
    {"driver": "Charles Leclerc", "lapTime": "1:17.259", "lap": 62, "avgSpeed": "216.999 km/h"},
    {"driver": "Nico Hulkenberg", "lapTime": "1:17.575", "lap": 63, "avgSpeed": "216.116 km/h"},
    {"driver": "Lewis Hamilton", "lapTime": "1:17.706", "lap": 62, "avgSpeed": "215.751 km/h"},
    {"driver": "Isack Hadjar", "lapTime": "1:17.770", "lap": 63, "avgSpeed": "215.574 km/h"},
    {"driver": "Pierre Gasly", "lapTime": "1:17.896", "lap": 63, "avgSpeed": "215.225 km/h"},
    {"driver": "Yuki Tsunoda", "lapTime": "1:17.998", "lap": 47, "avgSpeed": "214.943 km/h"},
    {"driver": "Fernando Alonso", "lapTime": "1:18.128", "lap": 66, "avgSpeed": "214.586 km/h"},
    {"driver": "Kimi Antonelli", "lapTime": "1:18.255", "lap": 52, "avgSpeed": "214.238 km/h"},
    {"driver": "Gabriel Bortoleto", "lapTime": "1:18.297", "lap": 52, "avgSpeed": "214.123 km/h"},
    {"driver": "Franco Colapinto", "lapTime": "1:18.353", "lap": 42, "avgSpeed": "213.970 km/h"},
    {"driver": "Esteban Ocon", "lapTime": "1:18.624", "lap": 47, "avgSpeed": "213.232 km/h"},
    {"driver": "Oliver Bearman", "lapTime": "1:18.907", "lap": 63, "avgSpeed": "212.467 km/h"},
    {"driver": "Carlos Sainz", "lapTime": "1:19.317", "lap": 65, "avgSpeed": "211.369 km/h"},
    {"driver": "Liam Lawson", "lapTime": "1:19.424", "lap": 62, "avgSpeed": "211.084 km/h"},
    {"driver": "Alexander Albon", "lapTime": "1:20.508", "lap": 9, "avgSpeed": "208.242 km/h"}
  ]
}, 
"monaco": {
  "round": 8,
  "name": "Monaco GP",
  "date": "May 23-25, 2025",
  "track": "Circuit de Monaco",
  "trackLength": 3.337,
  "weather": "Dry",
  "safetyCars": 0,
  "winner": {
    "driver": "Lando Norris",
    "team": "McLaren"
  },
  "fastestLap": {
    "driver": "Lando Norris",
    "time": "1:13.221",
    "lap": 78
  },
  "results": {
    "qualifying": [
      {"pos": 1, "driver": "Lando Norris", "team": "McLaren", "q1": "1:11.285", "q2": "1:10.570", "q3": "1:09.954"},
      {"pos": 2, "driver": "Charles Leclerc", "team": "Ferrari", "q1": "1:11.229", "q2": "1:10.581", "q3": "1:10.063"},
      {"pos": 3, "driver": "Oscar Piastri", "team": "McLaren", "q1": "1:11.308", "q2": "1:10.858", "q3": "1:10.129"},
      {"pos": 4, "driver": "Lewis Hamilton", "team": "Ferrari", "q1": "1:11.575", "q2": "1:10.883", "q3": "1:10.382"},
      {"pos": 5, "driver": "Max Verstappen", "team": "Red Bull", "q1": "1:11.431", "q2": "1:10.875", "q3": "1:10.669"},
      {"pos": 6, "driver": "Isack Hadjar", "team": "RB", "q1": "1:11.811", "q2": "1:11.040", "q3": "1:10.923"},
      {"pos": 7, "driver": "Fernando Alonso", "team": "Aston Martin", "q1": "1:11.674", "q2": "1:11.182", "q3": "1:10.924"},
      {"pos": 8, "driver": "Esteban Ocon", "team": "Haas", "q1": "1:11.839", "q2": "1:11.262", "q3": "1:10.942"},
      {"pos": 9, "driver": "Liam Lawson", "team": "RB", "q1": "1:11.818", "q2": "1:11.250", "q3": "1:11.129"},
      {"pos": 10, "driver": "Alexander Albon", "team": "Williams", "q1": "1:11.629", "q2": "1:10.732", "q3": "1:11.213"},
      {"pos": 11, "driver": "Carlos Sainz", "team": "Williams", "q1": "1:11.707", "q2": "1:11.362", "q3": null},
      {"pos": 12, "driver": "Yuki Tsunoda", "team": "Red Bull", "q1": "1:11.800", "q2": "1:11.415", "q3": null},
      {"pos": 13, "driver": "Nico Hulkenberg", "team": "Sauber", "q1": "1:11.871", "q2": "1:11.596", "q3": null},
      {"pos": 14, "driver": "George Russell", "team": "Mercedes", "q1": "1:11.507", "q2": null, "q3": null},
      {"pos": 15, "driver": "Kimi Antonelli", "team": "Mercedes", "q1": "1:11.880", "q2": null, "q3": null},
      {"pos": 16, "driver": "Gabriel Bortoleto", "team": "Sauber", "q1": "1:11.902", "q2": null, "q3": null},
      {"pos": 17, "driver": "Oliver Bearman", "team": "Haas", "q1": "1:11.979", "q2": null, "q3": null},
      {"pos": 18, "driver": "Pierre Gasly", "team": "Alpine", "q1": "1:11.994", "q2": null, "q3": null},
      {"pos": 19, "driver": "Lance Stroll", "team": "Aston Martin", "q1": "1:12.563", "q2": null, "q3": null},
      {"pos": 20, "driver": "Franco Colapinto", "team": "Alpine", "q1": "1:12.597", "q2": null, "q3": null}
    ],
    "race": [
      {"pos": 1, "driver": "Lando Norris", "team": "McLaren", "start": 1, "laps": 78, "points": 25},
      {"pos": 2, "driver": "Charles Leclerc", "team": "Ferrari", "start": 2, "laps": 78, "points": 18},
      {"pos": 3, "driver": "Oscar Piastri", "team": "McLaren", "start": 3, "laps": 78, "points": 15},
      {"pos": 4, "driver": "Max Verstappen", "team": "Red Bull", "start": 4, "laps": 78, "points": 12},
      {"pos": 5, "driver": "Lewis Hamilton", "team": "Ferrari", "start": 7, "laps": 78, "points": 10},
      {"pos": 6, "driver": "Isack Hadjar", "team": "RB", "start": 5, "laps": 77, "points": 8},
      {"pos": 7, "driver": "Esteban Ocon", "team": "Haas", "start": 8, "laps": 77, "points": 6},
      {"pos": 8, "driver": "Liam Lawson", "team": "RB", "start": 9, "laps": 77, "points": 4},
      {"pos": 9, "driver": "Alexander Albon", "team": "Williams", "start": 10, "laps": 76, "points": 2},
      {"pos": 10, "driver": "Carlos Sainz", "team": "Williams", "start": 11, "laps": 76, "points": 1},
      {"pos": 11, "driver": "George Russell", "team": "Mercedes", "start": 14, "laps": 76, "points": 0},
      {"pos": 12, "driver": "Oliver Bearman", "team": "Haas", "start": 20, "laps": 76, "points": 0},
      {"pos": 13, "driver": "Franco Colapinto", "team": "Alpine", "start": 18, "laps": 76, "points": 0},
      {"pos": 14, "driver": "Gabriel Bortoleto", "team": "Sauber", "start": 16, "laps": 76, "points": 0},
      {"pos": 15, "driver": "Lance Stroll", "team": "Aston Martin", "start": 19, "laps": 76, "points": 0},
      {"pos": 16, "driver": "Nico Hulkenberg", "team": "Sauber", "start": 13, "laps": 76, "points": 0},
      {"pos": 17, "driver": "Yuki Tsunoda", "team": "Red Bull", "start": 12, "laps": 76, "points": 0},
      {"pos": 18, "driver": "Kimi Antonelli", "team": "Mercedes", "start": 15, "laps": 75, "points": 0},
      {"pos": 19, "driver": "Fernando Alonso", "team": "Aston Martin", "start": 6, "laps": 36, "points": 0},
      {"pos": 20, "driver": "Pierre Gasly", "team": "Alpine", "start": 17, "laps": 7, "points": 0}
    ]
  },
  "highlights": [
    "Lando Norris wins dramatic Monaco GP after holding off Leclerc",
    "McLaren double podium with Piastri in 3rd",
    "Hamilton recovers to 5th after grid penalty",
    "Bearman makes up 8 positions from pit lane start",
    "Alonso DNF after collision with barrier at Sainte Devote"
  ],
  "practice": {
    "p1": [
      {"driver": "Charles Leclerc", "team": "Ferrari", "time": "1:11.964", "laps": 33},
      {"driver": "Max Verstappen", "team": "Red Bull", "time": "1:12.127", "laps": 30},
      {"driver": "Lando Norris", "team": "McLaren", "time": "1:12.290", "laps": 32},
      {"driver": "Alexander Albon", "team": "Williams", "time": "1:12.314", "laps": 33},
      {"driver": "Oscar Piastri", "team": "McLaren", "time": "1:12.342", "laps": 28},
      {"driver": "George Russell", "team": "Mercedes", "time": "1:12.482", "laps": 33},
      {"driver": "Carlos Sainz", "team": "Williams", "time": "1:12.534", "laps": 36},
      {"driver": "Pierre Gasly", "team": "Alpine", "time": "1:12.669", "laps": 29},
      {"driver": "Lewis Hamilton", "team": "Ferrari", "time": "1:12.690", "laps": 30},
      {"driver": "Fernando Alonso", "team": "Aston Martin", "time": "1:12.727", "laps": 28},
      {"driver": "Kimi Antonelli", "team": "Mercedes", "time": "1:12.765", "laps": 34},
      {"driver": "Nico Hulkenberg", "team": "Sauber", "time": "1:12.979", "laps": 30},
      {"driver": "Isack Hadjar", "team": "RB", "time": "1:13.187", "laps": 35},
      {"driver": "Yuki Tsunoda", "team": "Red Bull", "time": "1:13.232", "laps": 32},
      {"driver": "Oliver Bearman", "team": "Haas", "time": "1:13.329", "laps": 34},
      {"driver": "Esteban Ocon", "team": "Haas", "time": "1:13.394", "laps": 31},
      {"driver": "Liam Lawson", "team": "RB", "time": "1:13.429", "laps": 37},
      {"driver": "Gabriel Bortoleto", "team": "Sauber", "time": "1:13.470", "laps": 29},
      {"driver": "Franco Colapinto", "team": "Alpine", "time": "1:13.820", "laps": 32},
      {"driver": "Lance Stroll", "team": "Aston Martin", "time": "1:15.635", "laps": 4}
    ],
    "p2": [
      {"driver": "Charles Leclerc", "team": "Ferrari", "time": "1:11.355", "laps": 32},
      {"driver": "Oscar Piastri", "team": "McLaren", "time": "1:11.393", "laps": 28},
      {"driver": "Lewis Hamilton", "team": "Ferrari", "time": "1:11.460", "laps": 30},
      {"driver": "Lando Norris", "team": "McLaren", "time": "1:11.677", "laps": 32},
      {"driver": "Liam Lawson", "team": "RB", "time": "1:11.823", "laps": 32},
      {"driver": "Isack Hadjar", "team": "RB", "time": "1:11.842", "laps": 17},
      {"driver": "Fernando Alonso", "team": "Aston Martin", "time": "1:11.890", "laps": 30},
      {"driver": "Alexander Albon", "team": "Williams", "time": "1:11.918", "laps": 34},
      {"driver": "Kimi Antonelli", "team": "Mercedes", "time": "1:12.002", "laps": 32},
      {"driver": "Max Verstappen", "team": "Red Bull", "time": "1:12.068", "laps": 29},
      {"driver": "Yuki Tsunoda", "team": "Red Bull", "time": "1:12.072", "laps": 30},
      {"driver": "George Russell", "team": "Mercedes", "time": "1:12.092", "laps": 32},
      {"driver": "Carlos Sainz", "team": "Williams", "time": "1:12.151", "laps": 32},
      {"driver": "Gabriel Bortoleto", "team": "Sauber", "time": "1:12.234", "laps": 29},
      {"driver": "Oliver Bearman", "team": "Haas", "time": "1:12.259", "laps": 33},
      {"driver": "Nico Hulkenberg", "team": "Sauber", "time": "1:12.262", "laps": 32},
      {"driver": "Pierre Gasly", "team": "Alpine", "time": "1:12.404", "laps": 34},
      {"driver": "Lance Stroll", "team": "Aston Martin", "time": "1:12.512", "laps": 30},
      {"driver": "Esteban Ocon", "team": "Haas", "time": "1:12.541", "laps": 33},
      {"driver": "Franco Colapinto", "team": "Alpine", "time": "1:13.415", "laps": 31}
    ],
    "p3": [
      {"driver": "Charles Leclerc", "team": "Ferrari", "time": "1:10.953", "laps": 25},
      {"driver": "Max Verstappen", "team": "Red Bull", "time": "1:11.233", "laps": 23},
      {"driver": "Lando Norris", "team": "McLaren", "time": "1:11.247", "laps": 23},
      {"driver": "Oscar Piastri", "team": "McLaren", "time": "1:11.398", "laps": 21},
      {"driver": "Lewis Hamilton", "team": "Ferrari", "time": "1:11.516", "laps": 26},
      {"driver": "Alexander Albon", "team": "Williams", "time": "1:11.668", "laps": 24},
      {"driver": "Liam Lawson", "team": "RB", "time": "1:11.814", "laps": 33},
      {"driver": "Carlos Sainz", "team": "Williams", "time": "1:11.893", "laps": 24},
      {"driver": "Yuki Tsunoda", "team": "Red Bull", "time": "1:11.952", "laps": 21},
      {"driver": "Kimi Antonelli", "team": "Mercedes", "time": "1:12.013", "laps": 23},
      {"driver": "George Russell", "team": "Mercedes", "time": "1:12.066", "laps": 22},
      {"driver": "Fernando Alonso", "team": "Aston Martin", "time": "1:12.101", "laps": 22},
      {"driver": "Nico Hulkenberg", "team": "Sauber", "time": "1:12.125", "laps": 25},
      {"driver": "Pierre Gasly", "team": "Alpine", "time": "1:12.194", "laps": 20},
      {"driver": "Lance Stroll", "team": "Aston Martin", "time": "1:12.202", "laps": 24},
      {"driver": "Oliver Bearman", "team": "Haas", "time": "1:12.251", "laps": 20},
      {"driver": "Isack Hadjar", "team": "RB", "time": "1:12.271", "laps": 30},
      {"driver": "Esteban Ocon", "team": "Haas", "time": "1:12.499", "laps": 25},
      {"driver": "Gabriel Bortoleto", "team": "Sauber", "time": "1:12.601", "laps": 28},
      {"driver": "Franco Colapinto", "team": "Alpine", "time": "1:12.851", "laps": 32}
    ]
  },
  "pitStops": [
    {
      "driver": "Max Verstappen",
      "stops": 2,
      "totalTime": 48.064,
      "stopsDetail": [
        {"lap": 28, "time": 24.114},
        {"lap": 77, "time": 23.950}
      ]
    },
    {
      "driver": "Lando Norris",
      "stops": 2,
      "totalTime": 48.249,
      "stopsDetail": [
        {"lap": 19, "time": 24.545},
        {"lap": 50, "time": 23.704}
      ]
    },
    {
      "driver": "Charles Leclerc",
      "stops": 2,
      "totalTime": 47.254,
      "stopsDetail": [
        {"lap": 22, "time": 23.576},
        {"lap": 49, "time": 23.678}
      ]
    },
    {
      "driver": "Oscar Piastri",
      "stops": 2,
      "totalTime": 49.664,
      "stopsDetail": [
        {"lap": 20, "time": 25.756},
        {"lap": 48, "time": 23.908}
      ]
    },
    {
      "driver": "Lewis Hamilton",
      "stops": 2,
      "totalTime": 47.821,
      "stopsDetail": [
        {"lap": 18, "time": 23.565},
        {"lap": 56, "time": 24.256}
      ]
    },
    {
      "driver": "Isack Hadjar",
      "stops": 2,
      "totalTime": 48.342,
      "stopsDetail": [
        {"lap": 14, "time": 24.311},
        {"lap": 19, "time": 24.031}
      ]
    },
    {
      "driver": "Esteban Ocon",
      "stops": 2,
      "totalTime": 50.348,
      "stopsDetail": [
        {"lap": 16, "time": 25.575},
        {"lap": 28, "time": 24.773}
      ]
    },
    {
      "driver": "Liam Lawson",
      "stops": 2,
      "totalTime": 48.526,
      "stopsDetail": [
        {"lap": 31, "time": 24.206},
        {"lap": 40, "time": 24.320}
      ]
    },
    {
      "driver": "Alexander Albon",
      "stops": 2,
      "totalTime": 48.117,
      "stopsDetail": [
        {"lap": 32, "time": 23.927},
        {"lap": 40, "time": 24.190}
      ]
    },
    {
      "driver": "Carlos Sainz",
      "stops": 2,
      "totalTime": 48.547,
      "stopsDetail": [
        {"lap": 48, "time": 24.498},
        {"lap": 53, "time": 24.049}
      ]
    },
    {
      "driver": "George Russell",
      "stops": 3,
      "totalTime": 68.570,
      "stopsDetail": [
        {"lap": 62, "time": 25.063},
        {"lap": 68, "time": 24.025},
        {"lap": 69, "time": 19.482}
      ]
    },
    {
      "driver": "Oliver Bearman",
      "stops": 2,
      "totalTime": 74.771,
      "stopsDetail": [
        {"lap": 1, "time": 50.191},
        {"lap": 17, "time": 24.580}
      ]
    },
    {
      "driver": "Franco Colapinto",
      "stops": 2,
      "totalTime": 48.852,
      "stopsDetail": [
        {"lap": 13, "time": 24.355},
        {"lap": 26, "time": 24.497}
      ]
    },
    {
      "driver": "Gabriel Bortoleto",
      "stops": 3,
      "totalTime": 85.972,
      "stopsDetail": [
        {"lap": 1, "time": 37.243},
        {"lap": 26, "time": 24.483},
        {"lap": 35, "time": 24.246}
      ]
    },
    {
      "driver": "Lance Stroll",
      "stops": 2,
      "totalTime": 49.894,
      "stopsDetail": [
        {"lap": 17, "time": 23.978},
        {"lap": 64, "time": 25.916}
      ]
    },
    {
      "driver": "Nico Hulkenberg",
      "stops": 2,
      "totalTime": 48.462,
      "stopsDetail": [
        {"lap": 12, "time": 24.065},
        {"lap": 44, "time": 24.397}
      ]
    },
    {
      "driver": "Yuki Tsunoda",
      "stops": 2,
      "totalTime": 48.611,
      "stopsDetail": [
        {"lap": 1, "time": 24.525},
        {"lap": 73, "time": 24.086}
      ]
    },
    {
      "driver": "Kimi Antonelli",
      "stops": 2,
      "totalTime": 49.294,
      "stopsDetail": [
        {"lap": 69, "time": 24.457},
        {"lap": 71, "time": 24.837}
      ]
    },
    {
      "driver": "Fernando Alonso",
      "stops": 1,
      "totalTime": 25.375,
      "stopsDetail": [
        {"lap": 16, "time": 25.375}
      ]
    },
    {
      "driver": "Pierre Gasly",
      "stops": 1,
      "totalTime": 24.376,
      "stopsDetail": [
        {"lap": 1, "time": 24.376}
      ]
    }
  ],
  "fastestLaps": [
    {"driver": "Lando Norris", "lapTime": "1:13.221", "lap": 78, "avgSpeed": "164.067 km/h"},
    {"driver": "George Russell", "lapTime": "1:13.405", "lap": 76, "avgSpeed": "163.656 km/h"},
    {"driver": "Kimi Antonelli", "lapTime": "1:13.518", "lap": 77, "avgSpeed": "163.404 km/h"},
    {"driver": "Oscar Piastri", "lapTime": "1:13.745", "lap": 60, "avgSpeed": "162.901 km/h"},
    {"driver": "Carlos Sainz", "lapTime": "1:13.988", "lap": 70, "avgSpeed": "162.366 km/h"},
    {"driver": "Charles Leclerc", "lapTime": "1:14.055", "lap": 36, "avgSpeed": "162.219 km/h"},
    {"driver": "Lewis Hamilton", "lapTime": "1:14.090", "lap": 73, "avgSpeed": "162.143 km/h"},
    {"driver": "Max Verstappen", "lapTime": "1:14.230", "lap": 45, "avgSpeed": "161.837 km/h"},
    {"driver": "Alexander Albon", "lapTime": "1:14.597", "lap": 76, "avgSpeed": "161.041 km/h"},
    {"driver": "Oliver Bearman", "lapTime": "1:14.855", "lap": 6, "avgSpeed": "160.486 km/h"},
    {"driver": "Lance Stroll", "lapTime": "1:14.877", "lap": 69, "avgSpeed": "160.439 km/h"},
    {"driver": "Gabriel Bortoleto", "lapTime": "1:14.884", "lap": 38, "avgSpeed": "160.424 km/h"},
    {"driver": "Yuki Tsunoda", "lapTime": "1:14.913", "lap": 77, "avgSpeed": "160.362 km/h"},
    {"driver": "Esteban Ocon", "lapTime": "1:15.157", "lap": 34, "avgSpeed": "159.841 km/h"},
    {"driver": "Nico Hulkenberg", "lapTime": "1:15.223", "lap": 49, "avgSpeed": "159.701 km/h"},
    {"driver": "Franco Colapinto", "lapTime": "1:15.298", "lap": 31, "avgSpeed": "159.542 km/h"},
    {"driver": "Liam Lawson", "lapTime": "1:15.321", "lap": 55, "avgSpeed": "159.493 km/h"},
    {"driver": "Fernando Alonso", "lapTime": "1:15.593", "lap": 15, "avgSpeed": "158.919 km/h"},
    {"driver": "Isack Hadjar", "lapTime": "1:15.981", "lap": 16, "avgSpeed": "158.107 km/h"},
    {"driver": "Pierre Gasly", "lapTime": "1:18.054", "lap": 6, "avgSpeed": "153.908 km/h"}
  ]
},
"imola": {
  "round": 7,
  "name": "Emilia Romagna GP",
  "date": "May 16-18, 2025",
  "track": "Autodromo Internazionale Enzo e Dino Ferrari",
  "trackLength": 4.909,
  "weather": "Dry",
  "safetyCars": 0,
  "winner": {
    "driver": "Max Verstappen",
    "team": "Red Bull"
  },
  "fastestLap": {
    "driver": "Max Verstappen",
    "time": "1:17.988",
    "lap": 58
  },
  "results": {
    "qualifying": [
      {"pos": 1, "driver": "Oscar Piastri", "team": "McLaren", "q1": "1:15.500", "q2": "1:15.214", "q3": "1:14.670"},
      {"pos": 2, "driver": "Max Verstappen", "team": "Red Bull", "q1": "1:15.175", "q2": "1:15.394", "q3": "1:14.704"},
      {"pos": 3, "driver": "George Russell", "team": "Mercedes", "q1": "1:15.852", "q2": "1:15.334", "q3": "1:14.807"},
      {"pos": 4, "driver": "Lando Norris", "team": "McLaren", "q1": "1:15.894", "q2": "1:15.261", "q3": "1:14.962"},
      {"pos": 5, "driver": "Fernando Alonso", "team": "Aston Martin", "q1": "1:15.695", "q2": "1:15.442", "q3": "1:15.431"},
      {"pos": 6, "driver": "Carlos Sainz", "team": "Williams", "q1": "1:15.987", "q2": "1:15.198", "q3": "1:15.432"},
      {"pos": 7, "driver": "Alexander Albon", "team": "Williams", "q1": "1:16.123", "q2": "1:15.521", "q3": "1:15.473"},
      {"pos": 8, "driver": "Lance Stroll", "team": "Aston Martin", "q1": "1:15.817", "q2": "1:15.497", "q3": "1:15.581"},
      {"pos": 9, "driver": "Isack Hadjar", "team": "RB", "q1": "1:16.253", "q2": "1:15.510", "q3": "1:15.746"},
      {"pos": 10, "driver": "Pierre Gasly", "team": "Alpine", "q1": "1:15.937", "q2": "1:15.505", "q3": "1:15.787"},
      {"pos": 11, "driver": "Charles Leclerc", "team": "Ferrari", "q1": "1:16.108", "q2": "1:15.604", "q3": null},
      {"pos": 12, "driver": "Lewis Hamilton", "team": "Ferrari", "q1": "1:16.163", "q2": "1:15.765", "q3": null},
      {"pos": 13, "driver": "Kimi Antonelli", "team": "Mercedes", "q1": "1:15.943", "q2": "1:15.772", "q3": null},
      {"pos": 14, "driver": "Gabriel Bortoleto", "team": "Sauber", "q1": "1:16.340", "q2": "1:16.260", "q3": null},
      {"pos": 15, "driver": "Franco Colapinto", "team": "Alpine", "q1": "1:16.256", "q2": null, "q3": null},
      {"pos": 16, "driver": "Liam Lawson", "team": "RB", "q1": "1:16.379", "q2": null, "q3": null},
      {"pos": 17, "driver": "Nico Hulkenberg", "team": "Sauber", "q1": "1:16.518", "q2": null, "q3": null},
      {"pos": 18, "driver": "Esteban Ocon", "team": "Haas", "q1": "1:16.613", "q2": null, "q3": null},
      {"pos": 19, "driver": "Oliver Bearman", "team": "Haas", "q1": "1:16.918", "q2": null, "q3": null},
      {"pos": 20, "driver": "Yuki Tsunoda", "team": "Red Bull", "q1": null, "q2": null, "q3": null}
    ],
    "race": [
      {"pos": 1, "driver": "Max Verstappen", "team": "Red Bull", "start": 2, "laps": 63, "points": 25},
      {"pos": 2, "driver": "Lando Norris", "team": "McLaren", "start": 4, "laps": 63, "points": 18},
      {"pos": 3, "driver": "Oscar Piastri", "team": "McLaren", "start": 1, "laps": 63, "points": 15},
      {"pos": 4, "driver": "Lewis Hamilton", "team": "Ferrari", "start": 12, "laps": 63, "points": 12},
      {"pos": 5, "driver": "Alexander Albon", "team": "Williams", "start": 7, "laps": 63, "points": 10},
      {"pos": 6, "driver": "Charles Leclerc", "team": "Ferrari", "start": 11, "laps": 63, "points": 8},
      {"pos": 7, "driver": "George Russell", "team": "Mercedes", "start": 3, "laps": 63, "points": 6},
      {"pos": 8, "driver": "Carlos Sainz", "team": "Williams", "start": 6, "laps": 63, "points": 4},
      {"pos": 9, "driver": "Isack Hadjar", "team": "RB", "start": 9, "laps": 63, "points": 2},
      {"pos": 10, "driver": "Yuki Tsunoda", "team": "Red Bull", "start": 20, "laps": 63, "points": 1},
      {"pos": 11, "driver": "Fernando Alonso", "team": "Aston Martin", "start": 5, "laps": 63, "points": 0},
      {"pos": 12, "driver": "Nico Hulkenberg", "team": "Sauber", "start": 17, "laps": 63, "points": 0},
      {"pos": 13, "driver": "Pierre Gasly", "team": "Alpine", "start": 10, "laps": 63, "points": 0},
      {"pos": 14, "driver": "Liam Lawson", "team": "RB", "start": 15, "laps": 63, "points": 0},
      {"pos": 15, "driver": "Lance Stroll", "team": "Aston Martin", "start": 8, "laps": 63, "points": 0},
      {"pos": 16, "driver": "Franco Colapinto", "team": "Alpine", "start": 16, "laps": 63, "points": 0},
      {"pos": 17, "driver": "Oliver Bearman", "team": "Haas", "start": 19, "laps": 63, "points": 0},
      {"pos": 18, "driver": "Gabriel Bortoleto", "team": "Sauber", "start": 14, "laps": 63, "points": 0},
      {"pos": 19, "driver": "Kimi Antonelli", "team": "Mercedes", "start": 13, "laps": 44, "points": 0},
      {"pos": 20, "driver": "Esteban Ocon", "team": "Haas", "start": 18, "laps": 27, "points": 0}
    ]
  },
  "highlights": [
    "Verstappen wins after thrilling battle with McLaren duo",
    "Hamilton recovers from P12 to finish 4th",
    "Tsunoda scores point after starting from pit lane",
    "McLaren double podium with Piastri P3",
    "Antonelli and Ocon retire with technical issues"
  ],
  "practice": {
    "p1": [
      {"driver": "Oscar Piastri", "team": "McLaren", "time": "1:16.545", "laps": 23},
      {"driver": "Lando Norris", "team": "McLaren", "time": "1:16.577", "laps": 23},
      {"driver": "Carlos Sainz", "team": "Williams", "time": "1:16.597", "laps": 19},
      {"driver": "George Russell", "team": "Mercedes", "time": "1:16.599", "laps": 26},
      {"driver": "Lewis Hamilton", "team": "Ferrari", "time": "1:16.641", "laps": 22},
      {"driver": "Pierre Gasly", "team": "Alpine", "time": "1:16.696", "laps": 21},
      {"driver": "Max Verstappen", "team": "Red Bull", "time": "1:16.905", "laps": 15},
      {"driver": "Alexander Albon", "team": "Williams", "time": "1:16.922", "laps": 21},
      {"driver": "Gabriel Bortoleto", "team": "Sauber", "time": "1:16.925", "laps": 23},
      {"driver": "Nico Hulkenberg", "team": "Sauber", "time": "1:16.998", "laps": 23},
      {"driver": "Lance Stroll", "team": "Aston Martin", "time": "1:17.032", "laps": 23},
      {"driver": "Charles Leclerc", "team": "Ferrari", "time": "1:17.077", "laps": 23},
      {"driver": "Kimi Antonelli", "team": "Mercedes", "time": "1:17.094", "laps": 23},
      {"driver": "Fernando Alonso", "team": "Aston Martin", "time": "1:17.121", "laps": 22},
      {"driver": "Liam Lawson", "team": "RB", "time": "1:17.286", "laps": 22},
      {"driver": "Yuki Tsunoda", "team": "Red Bull", "time": "1:17.356", "laps": 18},
      {"driver": "Franco Colapinto", "team": "Alpine", "time": "1:17.373", "laps": 21},
      {"driver": "Oliver Bearman", "team": "Haas", "time": "1:17.446", "laps": 20},
      {"driver": "Isack Hadjar", "team": "RB", "time": "1:17.641", "laps": 19},
      {"driver": "Esteban Ocon", "team": "Haas", "time": "1:17.662", "laps": 20}
    ],
    "p2": [
      {"driver": "Oscar Piastri", "team": "McLaren", "time": "1:15.293", "laps": 28},
      {"driver": "Lando Norris", "team": "McLaren", "time": "1:15.318", "laps": 25},
      {"driver": "Pierre Gasly", "team": "Alpine", "time": "1:15.569", "laps": 29},
      {"driver": "George Russell", "team": "Mercedes", "time": "1:15.693", "laps": 23},
      {"driver": "Max Verstappen", "team": "Red Bull", "time": "1:15.735", "laps": 26},
      {"driver": "Charles Leclerc", "team": "Ferrari", "time": "1:15.768", "laps": 28},
      {"driver": "Isack Hadjar", "team": "RB", "time": "1:15.792", "laps": 22},
      {"driver": "Yuki Tsunoda", "team": "Red Bull", "time": "1:15.827", "laps": 29},
      {"driver": "Alexander Albon", "team": "Williams", "time": "1:15.916", "laps": 29},
      {"driver": "Carlos Sainz", "team": "Williams", "time": "1:15.934", "laps": 29},
      {"driver": "Lewis Hamilton", "team": "Ferrari", "time": "1:15.943", "laps": 26},
      {"driver": "Oliver Bearman", "team": "Haas", "time": "1:16.009", "laps": 26},
      {"driver": "Franco Colapinto", "team": "Alpine", "time": "1:16.044", "laps": 29},
      {"driver": "Fernando Alonso", "team": "Aston Martin", "time": "1:16.220", "laps": 21},
      {"driver": "Liam Lawson", "team": "RB", "time": "1:16.255", "laps": 22},
      {"driver": "Gabriel Bortoleto", "team": "Sauber", "time": "1:16.339", "laps": 27},
      {"driver": "Lance Stroll", "team": "Aston Martin", "time": "1:16.341", "laps": 23},
      {"driver": "Kimi Antonelli", "team": "Mercedes", "time": "1:16.406", "laps": 27},
      {"driver": "Nico Hulkenberg", "team": "Sauber", "time": "1:16.419", "laps": 26},
      {"driver": "Esteban Ocon", "team": "Haas", "time": "1:16.420", "laps": 24}
    ],
    "p3": [
      {"driver": "Lando Norris", "team": "McLaren", "time": "1:14.897", "laps": 19},
      {"driver": "Oscar Piastri", "team": "McLaren", "time": "1:14.997", "laps": 19},
      {"driver": "Max Verstappen", "team": "Red Bull", "time": "1:15.078", "laps": 17},
      {"driver": "Kimi Antonelli", "team": "Mercedes", "time": "1:15.399", "laps": 15},
      {"driver": "Charles Leclerc", "team": "Ferrari", "time": "1:15.451", "laps": 26},
      {"driver": "Carlos Sainz", "team": "Williams", "time": "1:15.457", "laps": 21},
      {"driver": "Isack Hadjar", "team": "RB", "time": "1:15.508", "laps": 20},
      {"driver": "George Russell", "team": "Mercedes", "time": "1:15.662", "laps": 14},
      {"driver": "Alexander Albon", "team": "Williams", "time": "1:15.732", "laps": 20},
      {"driver": "Lewis Hamilton", "team": "Ferrari", "time": "1:15.787", "laps": 25},
      {"driver": "Fernando Alonso", "team": "Aston Martin", "time": "1:15.819", "laps": 18},
      {"driver": "Oliver Bearman", "team": "Haas", "time": "1:15.944", "laps": 14},
      {"driver": "Lance Stroll", "team": "Aston Martin", "time": "1:15.975", "laps": 22},
      {"driver": "Liam Lawson", "team": "RB", "time": "1:15.977", "laps": 21},
      {"driver": "Pierre Gasly", "team": "Alpine", "time": "1:15.990", "laps": 14},
      {"driver": "Gabriel Bortoleto", "team": "Sauber", "time": "1:16.046", "laps": 17},
      {"driver": "Yuki Tsunoda", "team": "Red Bull", "time": "1:16.110", "laps": 14},
      {"driver": "Franco Colapinto", "team": "Alpine", "time": "1:16.210", "laps": 14},
      {"driver": "Nico Hulkenberg", "team": "Sauber", "time": "1:16.238", "laps": 16},
      {"driver": "Esteban Ocon", "team": "Haas", "time": "1:16.387", "laps": 15}
    ]
  },
  "pitStops": [
    {
      "driver": "Max Verstappen",
      "stops": 2,
      "totalTime": 59.648,
      "stopsDetail": [
        {"lap": 29, "time": 29.991},
        {"lap": 46, "time": 29.657}
      ]
    },
    {
      "driver": "Lando Norris",
      "stops": 2,
      "totalTime": 61.188,
      "stopsDetail": [
        {"lap": 28, "time": 29.513},
        {"lap": 46, "time": 31.675}
      ]
    },
    {
      "driver": "Oscar Piastri",
      "stops": 2,
      "totalTime": 61.582,
      "stopsDetail": [
        {"lap": 13, "time": 30.973},
        {"lap": 30, "time": 30.609}
      ]
    },
    {
      "driver": "Lewis Hamilton",
      "stops": 2,
      "totalTime": 59.899,
      "stopsDetail": [
        {"lap": 29, "time": 29.949},
        {"lap": 46, "time": 29.950}
      ]
    },
    {
      "driver": "Alexander Albon",
      "stops": 2,
      "totalTime": 60.257,
      "stopsDetail": [
        {"lap": 29, "time": 30.471},
        {"lap": 47, "time": 29.786}
      ]
    },
    {
      "driver": "Charles Leclerc",
      "stops": 2,
      "totalTime": 61.532,
      "stopsDetail": [
        {"lap": 10, "time": 29.634},
        {"lap": 29, "time": 31.898}
      ]
    },
    {
      "driver": "George Russell",
      "stops": 2,
      "totalTime": 60.538,
      "stopsDetail": [
        {"lap": 11, "time": 29.784},
        {"lap": 29, "time": 30.754}
      ]
    },
    {
      "driver": "Carlos Sainz",
      "stops": 2,
      "totalTime": 60.715,
      "stopsDetail": [
        {"lap": 11, "time": 30.080},
        {"lap": 29, "time": 30.635}
      ]
    },
    {
      "driver": "Isack Hadjar",
      "stops": 2,
      "totalTime": 60.219,
      "stopsDetail": [
        {"lap": 29, "time": 30.123},
        {"lap": 46, "time": 30.096}
      ]
    },
    {
      "driver": "Yuki Tsunoda",
      "stops": 1,
      "totalTime": 29.807,
      "stopsDetail": [
        {"lap": 29, "time": 29.807}
      ]
    },
    {
      "driver": "Fernando Alonso",
      "stops": 2,
      "totalTime": 61.049,
      "stopsDetail": [
        {"lap": 12, "time": 30.163},
        {"lap": 46, "time": 30.886}
      ]
    },
    {
      "driver": "Nico Hulkenberg",
      "stops": 1,
      "totalTime": 30.184,
      "stopsDetail": [
        {"lap": 29, "time": 30.184}
      ]
    },
    {
      "driver": "Pierre Gasly",
      "stops": 2,
      "totalTime": 63.491,
      "stopsDetail": [
        {"lap": 9, "time": 31.488},
        {"lap": 29, "time": 32.003}
      ]
    },
    {
      "driver": "Liam Lawson",
      "stops": 2,
      "totalTime": 59.986,
      "stopsDetail": [
        {"lap": 10, "time": 29.695},
        {"lap": 29, "time": 30.291}
      ]
    },
    {
      "driver": "Lance Stroll",
      "stops": 2,
      "totalTime": 65.714,
      "stopsDetail": [
        {"lap": 14, "time": 31.218},
        {"lap": 46, "time": 34.496}
      ]
    },
    {
      "driver": "Franco Colapinto",
      "stops": 2,
      "totalTime": 61.424,
      "stopsDetail": [
        {"lap": 22, "time": 30.317},
        {"lap": 46, "time": 31.107}
      ]
    },
    {
      "driver": "Oliver Bearman",
      "stops": 2,
      "totalTime": 88.171,
      "stopsDetail": [
        {"lap": 1, "time": 50.191},
        {"lap": 31, "time": 56.757}
      ]
    },
    {
      "driver": "Gabriel Bortoleto",
      "stops": 3,
      "totalTime": 91.428,
      "stopsDetail": [
        {"lap": 12, "time": 30.906},
        {"lap": 29, "time": 30.605},
        {"lap": 46, "time": 29.917}
      ]
    },
    {
      "driver": "Esteban Ocon",
      "stops": 1,
      "totalTime": 30.506,
      "stopsDetail": [
        {"lap": 1, "time": 30.506}
      ]
    }
  ],
  "fastestLaps": [
    {"driver": "Max Verstappen", "lapTime": "1:17.988", "lap": 58, "avgSpeed": "226.604 km/h"},
    {"driver": "Lewis Hamilton", "lapTime": "1:18.265", "lap": 61, "avgSpeed": "225.802 km/h"},
    {"driver": "Alexander Albon", "lapTime": "1:18.289", "lap": 63, "avgSpeed": "225.732 km/h"},
    {"driver": "Lando Norris", "lapTime": "1:18.311", "lap": 63, "avgSpeed": "225.669 km/h"},
    {"driver": "Oscar Piastri", "lapTime": "1:18.894", "lap": 56, "avgSpeed": "224.001 km/h"},
    {"driver": "Charles Leclerc", "lapTime": "1:19.048", "lap": 56, "avgSpeed": "223.565 km/h"},
    {"driver": "Isack Hadjar", "lapTime": "1:19.473", "lap": 60, "avgSpeed": "222.369 km/h"},
    {"driver": "Oliver Bearman", "lapTime": "1:19.521", "lap": 52, "avgSpeed": "222.235 km/h"},
    {"driver": "George Russell", "lapTime": "1:19.733", "lap": 55, "avgSpeed": "221.644 km/h"},
    {"driver": "Carlos Sainz", "lapTime": "1:19.836", "lap": 58, "avgSpeed": "221.358 km/h"},
    {"driver": "Fernando Alonso", "lapTime": "1:19.894", "lap": 61, "avgSpeed": "221.198 km/h"},
    {"driver": "Yuki Tsunoda", "lapTime": "1:20.039", "lap": 60, "avgSpeed": "220.797 km/h"},
    {"driver": "Franco Colapinto", "lapTime": "1:20.345", "lap": 57, "avgSpeed": "219.956 km/h"},
    {"driver": "Pierre Gasly", "lapTime": "1:20.398", "lap": 58, "avgSpeed": "219.811 km/h"},
    {"driver": "Nico Hulkenberg", "lapTime": "1:20.401", "lap": 62, "avgSpeed": "219.803 km/h"},
    {"driver": "Liam Lawson", "lapTime": "1:20.473", "lap": 60, "avgSpeed": "219.606 km/h"},
    {"driver": "Lance Stroll", "lapTime": "1:20.501", "lap": 58, "avgSpeed": "219.530 km/h"},
    {"driver": "Kimi Antonelli", "lapTime": "1:20.620", "lap": 33, "avgSpeed": "219.206 km/h"},
    {"driver": "Gabriel Bortoleto", "lapTime": "1:20.630", "lap": 57, "avgSpeed": "219.178 km/h"},
    {"driver": "Esteban Ocon", "lapTime": "1:21.413", "lap": 3, "avgSpeed": "217.070 km/h"}
  ]
},
"miami": {
  "round": 6,
  "name": "Miami Grand Prix",
  "date": "May 2-4, 2025",
  "track": "Miami International Autodrome",
  "trackLength": 5.412,
  "weather": "Dry",
  "safetyCars": 0,
  "winner": {
    "driver": "Oscar Piastri",
    "team": "McLaren"
  },
  "fastestLap": {
    "driver": "Lando Norris",
    "time": "1:29.746",
    "lap": 36
  },
  "results": {
    "practice": [
      {"pos": 1, "driver": "Oscar Piastri", "team": "McLaren", "time": "1:27.128", "laps": 22},
      {"pos": 2, "driver": "Charles Leclerc", "team": "Ferrari", "time": "1:27.484", "laps": 22},
      {"pos": 3, "driver": "Max Verstappen", "team": "Red Bull", "time": "1:27.558", "laps": 20},
      {"pos": 4, "driver": "Carlos Sainz", "team": "Williams", "time": "1:27.678", "laps": 23},
      {"pos": 5, "driver": "Alexander Albon", "team": "Williams", "time": "1:27.955", "laps": 25},
      {"pos": 6, "driver": "Isack Hadjar", "team": "RB", "time": "1:27.968", "laps": 23},
      {"pos": 7, "driver": "George Russell", "team": "Mercedes", "time": "1:28.058", "laps": 26},
      {"pos": 8, "driver": "Yuki Tsunoda", "team": "Red Bull", "time": "1:28.155", "laps": 21},
      {"pos": 9, "driver": "Kimi Antonelli", "team": "Mercedes", "time": "1:28.227", "laps": 28},
      {"pos": 10, "driver": "Fernando Alonso", "team": "Aston Martin", "time": "1:28.243", "laps": 24},
      {"pos": 11, "driver": "Liam Lawson", "team": "RB", "time": "1:28.374", "laps": 21},
      {"pos": 12, "driver": "Lando Norris", "team": "McLaren", "time": "1:28.391", "laps": 21},
      {"pos": 13, "driver": "Lewis Hamilton", "team": "Ferrari", "time": "1:28.556", "laps": 20},
      {"pos": 14, "driver": "Nico Hulkenberg", "team": "Sauber", "time": "1:28.573", "laps": 13},
      {"pos": 15, "driver": "Gabriel Bortoleto", "team": "Sauber", "time": "1:28.771", "laps": 19},
      {"pos": 16, "driver": "Oliver Bearman", "team": "Haas", "time": "1:28.996", "laps": 22},
      {"pos": 17, "driver": "Pierre Gasly", "team": "Alpine", "time": "1:29.084", "laps": 22},
      {"pos": 18, "driver": "Esteban Ocon", "team": "Haas", "time": "1:29.179", "laps": 19},
      {"pos": 19, "driver": "Jack Doohan", "team": "Alpine", "time": "1:29.357", "laps": 22},
      {"pos": 20, "driver": "Lance Stroll", "team": "Aston Martin", "time": "1:29.362", "laps": 21}
    ],
    "sprintQualifying": [
      {"pos": 1, "driver": "Kimi Antonelli", "team": "Mercedes", "q1": "1:27.858", "q2": "1:27.384", "q3": "1:26.482"},
      {"pos": 2, "driver": "Oscar Piastri", "team": "McLaren", "q1": "1:27.951", "q2": "1:27.354", "q3": "1:26.527"},
      {"pos": 3, "driver": "Lando Norris", "team": "McLaren", "q1": "1:27.890", "q2": "1:27.109", "q3": "1:26.582"},
      {"pos": 4, "driver": "Max Verstappen", "team": "Red Bull", "q1": "1:27.953", "q2": "1:27.245", "q3": "1:26.737"},
      {"pos": 5, "driver": "George Russell", "team": "Mercedes", "q1": "1:27.688", "q2": "1:27.666", "q3": "1:26.791"},
      {"pos": 6, "driver": "Charles Leclerc", "team": "Ferrari", "q1": "1:28.325", "q2": "1:27.467", "q3": "1:26.808"},
      {"pos": 7, "driver": "Lewis Hamilton", "team": "Ferrari", "q1": "1:28.231", "q2": "1:27.546", "q3": "1:27.030"},
      {"pos": 8, "driver": "Alexander Albon", "team": "Williams", "q1": "1:27.859", "q2": "1:27.697", "q3": "1:27.193"},
      {"pos": 9, "driver": "Isack Hadjar", "team": "RB", "q1": "1:28.394", "q2": "1:27.773", "q3": "1:27.543"},
      {"pos": 10, "driver": "Fernando Alonso", "team": "Aston Martin", "q1": "1:28.455", "q2": "1:27.766", "q3": "1:27.790"},
      {"pos": 11, "driver": "Nico Hulkenberg", "team": "Sauber", "q1": "1:28.542", "q2": "1:27.850", "q3": null},
      {"pos": 12, "driver": "Esteban Ocon", "team": "Haas", "q1": "1:28.303", "q2": "1:28.070", "q3": null},
      {"pos": 13, "driver": "Pierre Gasly", "team": "Alpine", "q1": "1:28.345", "q2": "1:28.167", "q3": null},
      {"pos": 14, "driver": "Liam Lawson", "team": "RB", "q1": "1:28.914", "q2": "1:28.375", "q3": null},
      {"pos": 15, "driver": "Carlos Sainz", "team": "Williams", "q1": "1:27.899", "q2": null, "q3": null},
      {"pos": 16, "driver": "Lance Stroll", "team": "Aston Martin", "q1": "1:29.028", "q2": null, "q3": null},
      {"pos": 17, "driver": "Jack Doohan", "team": "Alpine", "q1": "1:29.171", "q2": null, "q3": null},
      {"pos": 18, "driver": "Yuki Tsunoda", "team": "Red Bull", "q1": "1:29.246", "q2": null, "q3": null},
      {"pos": 19, "driver": "Gabriel Bortoleto", "team": "Sauber", "q1": "1:29.312", "q2": null, "q3": null},
      {"pos": 20, "driver": "Oliver Bearman", "team": "Haas", "q1": "1:29.825", "q2": null, "q3": null}
    ],
    "sprint": [
      {"pos": 1, "driver": "Lando Norris", "team": "McLaren", "start": 3, "laps": 18, "points": 8},
      {"pos": 2, "driver": "Oscar Piastri", "team": "McLaren", "start": 2, "laps": 18, "points": 7},
      {"pos": 3, "driver": "Lewis Hamilton", "team": "Ferrari", "start": 7, "laps": 18, "points": 6},
      {"pos": 4, "driver": "George Russell", "team": "Mercedes", "start": 5, "laps": 18, "points": 5},
      {"pos": 5, "driver": "Lance Stroll", "team": "Aston Martin", "start": 16, "laps": 18, "points": 4},
      {"pos": 6, "driver": "Yuki Tsunoda", "team": "Red Bull", "start": 20, "laps": 18, "points": 3},
      {"pos": 7, "driver": "Kimi Antonelli", "team": "Mercedes", "start": 1, "laps": 18, "points": 2},
      {"pos": 8, "driver": "Pierre Gasly", "team": "Alpine", "start": 13, "laps": 18, "points": 1},
      {"pos": 9, "driver": "Nico Hulkenberg", "team": "Sauber", "start": 11, "laps": 18, "points": 0},
      {"pos": 10, "driver": "Isack Hadjar", "team": "RB", "start": 9, "laps": 18, "points": 0},
      {"pos": 11, "driver": "Alexander Albon", "team": "Williams", "start": 8, "laps": 18, "points": 0},
      {"pos": 12, "driver": "Esteban Ocon", "team": "Haas", "start": 12, "laps": 18, "points": 0},
      {"pos": 13, "driver": "Liam Lawson", "team": "RB", "start": 14, "laps": 18, "points": 0},
      {"pos": 14, "driver": "Oliver Bearman", "team": "Haas", "start": 19, "laps": 18, "points": 0},
      {"pos": 15, "driver": "Gabriel Bortoleto", "team": "Sauber", "start": 18, "laps": 18, "points": 0},
      {"pos": 16, "driver": "Jack Doohan", "team": "Alpine", "start": 17, "laps": 18, "points": 0},
      {"pos": 17, "driver": "Max Verstappen", "team": "Red Bull", "start": 4, "laps": 18, "points": 0},
      {"pos": 18, "driver": "Fernando Alonso", "team": "Aston Martin", "start": 10, "laps": 13, "points": 0},
      {"pos": 19, "driver": "Carlos Sainz", "team": "Williams", "start": 15, "laps": 12, "points": 0},
      {"pos": 20, "driver": "Charles Leclerc", "team": "Ferrari", "start": 6, "laps": 0, "points": 0}
    ],
    "qualifying": [
      {"pos": 1, "driver": "Max Verstappen", "team": "Red Bull", "q1": "1:26.870", "q2": "1:26.643", "q3": "1:26.204"},
      {"pos": 2, "driver": "Lando Norris", "team": "McLaren", "q1": "1:26.955", "q2": "1:26.499", "q3": "1:26.269"},
      {"pos": 3, "driver": "Kimi Antonelli", "team": "Mercedes", "q1": "1:27.077", "q2": "1:26.606", "q3": "1:26.271"},
      {"pos": 4, "driver": "Oscar Piastri", "team": "McLaren", "q1": "1:27.006", "q2": "1:26.269", "q3": "1:26.375"},
      {"pos": 5, "driver": "George Russell", "team": "Mercedes", "q1": "1:27.014", "q2": "1:26.575", "q3": "1:26.385"},
      {"pos": 6, "driver": "Carlos Sainz", "team": "Williams", "q1": "1:27.098", "q2": "1:26.847", "q3": "1:26.569"},
      {"pos": 7, "driver": "Alexander Albon", "team": "Williams", "q1": "1:27.042", "q2": "1:26.855", "q3": "1:26.682"},
      {"pos": 8, "driver": "Charles Leclerc", "team": "Ferrari", "q1": "1:27.417", "q2": "1:26.948", "q3": "1:26.754"},
      {"pos": 9, "driver": "Esteban Ocon", "team": "Haas", "q1": "1:27.450", "q2": "1:26.967", "q3": "1:26.824"},
      {"pos": 10, "driver": "Yuki Tsunoda", "team": "Red Bull", "q1": "1:27.298", "q2": "1:26.959", "q3": "1:26.943"},
      {"pos": 11, "driver": "Isack Hadjar", "team": "RB", "q1": "1:27.301", "q2": "1:26.987", "q3": null},
      {"pos": 12, "driver": "Lewis Hamilton", "team": "Ferrari", "q1": "1:27.279", "q2": "1:27.006", "q3": null},
      {"pos": 13, "driver": "Gabriel Bortoleto", "team": "Sauber", "q1": "1:27.343", "q2": "1:27.151", "q3": null},
      {"pos": 14, "driver": "Jack Doohan", "team": "Alpine", "q1": "1:27.422", "q2": "1:27.186", "q3": null},
      {"pos": 15, "driver": "Liam Lawson", "team": "RB", "q1": "1:27.444", "q2": "1:27.363", "q3": null},
      {"pos": 16, "driver": "Nico Hulkenberg", "team": "Sauber", "q1": "1:27.473", "q2": null, "q3": null},
      {"pos": 17, "driver": "Fernando Alonso", "team": "Aston Martin", "q1": "1:27.604", "q2": null, "q3": null},
      {"pos": 18, "driver": "Pierre Gasly", "team": "Alpine", "q1": "1:27.710", "q2": null, "q3": null},
      {"pos": 19, "driver": "Lance Stroll", "team": "Aston Martin", "q1": "1:27.830", "q2": null, "q3": null},
      {"pos": 20, "driver": "Oliver Bearman", "team": "Haas", "q1": "1:27.999", "q2": null, "q3": null}
    ],
    "race": [
      {"pos": 1, "driver": "Oscar Piastri", "team": "McLaren", "start": 4, "laps": 57, "points": 25},
      {"pos": 2, "driver": "Lando Norris", "team": "McLaren", "start": 2, "laps": 57, "points": 18},
      {"pos": 3, "driver": "George Russell", "team": "Mercedes", "start": 5, "laps": 57, "points": 15},
      {"pos": 4, "driver": "Max Verstappen", "team": "Red Bull", "start": 1, "laps": 57, "points": 12},
      {"pos": 5, "driver": "Alexander Albon", "team": "Williams", "start": 7, "laps": 57, "points": 10},
      {"pos": 6, "driver": "Kimi Antonelli", "team": "Mercedes", "start": 3, "laps": 57, "points": 8},
      {"pos": 7, "driver": "Charles Leclerc", "team": "Ferrari", "start": 8, "laps": 57, "points": 6},
      {"pos": 8, "driver": "Lewis Hamilton", "team": "Ferrari", "start": 12, "laps": 57, "points": 4},
      {"pos": 9, "driver": "Carlos Sainz", "team": "Williams", "start": 6, "laps": 57, "points": 2},
      {"pos": 10, "driver": "Yuki Tsunoda", "team": "Red Bull", "start": 10, "laps": 57, "points": 1},
      {"pos": 11, "driver": "Isack Hadjar", "team": "RB", "start": 11, "laps": 57, "points": 0},
      {"pos": 12, "driver": "Esteban Ocon", "team": "Haas", "start": 9, "laps": 57, "points": 0},
      {"pos": 13, "driver": "Pierre Gasly", "team": "Alpine", "start": 20, "laps": 57, "points": 0},
      {"pos": 14, "driver": "Nico Hulkenberg", "team": "Sauber", "start": 16, "laps": 56, "points": 0},
      {"pos": 15, "driver": "Fernando Alonso", "team": "Aston Martin", "start": 17, "laps": 56, "points": 0},
      {"pos": 16, "driver": "Lance Stroll", "team": "Aston Martin", "start": 18, "laps": 56, "points": 0},
      {"pos": 17, "driver": "Liam Lawson", "team": "RB", "start": 15, "laps": 36, "points": 0},
      {"pos": 18, "driver": "Gabriel Bortoleto", "team": "Sauber", "start": 13, "laps": 30, "points": 0},
      {"pos": 19, "driver": "Oliver Bearman", "team": "Haas", "start": 19, "laps": 27, "points": 0},
      {"pos": 20, "driver": "Jack Doohan", "team": "Alpine", "start": 14, "laps": 0, "points": 0}
    ]
  },
  "highlights": [
    "McLaren 1-2 with Piastri taking his first win",
    "Norris completes McLaren double podium",
    "Russell holds off Verstappen for final podium spot",
    "Tsunoda scores point despite pit lane start",
    "Multiple DNFs including Doohan on first lap"
  ],
  "pitStops": [
    {
      "driver": "Gabriel Bortoleto",
      "stops": 1,
      "totalTime": 22.188,
      "stopsDetail": [{"lap": 19, "time": 22.188}]
    },
    {
      "driver": "Lance Stroll",
      "stops": 1,
      "totalTime": 22.317,
      "stopsDetail": [{"lap": 20, "time": 22.317}]
    },
    {
      "driver": "Isack Hadjar",
      "stops": 1,
      "totalTime": 22.115,
      "stopsDetail": [{"lap": 22, "time": 22.115}]
    },
    {
      "driver": "Esteban Ocon",
      "stops": 1,
      "totalTime": 23.162,
      "stopsDetail": [{"lap": 23, "time": 23.162}]
    },
    {
      "driver": "Kimi Antonelli",
      "stops": 1,
      "totalTime": 24.383,
      "stopsDetail": [{"lap": 25, "time": 24.383}]
    },
    {
      "driver": "Carlos Sainz",
      "stops": 1,
      "totalTime": 22.415,
      "stopsDetail": [{"lap": 25, "time": 22.415}]
    },
    {
      "driver": "Max Verstappen",
      "stops": 1,
      "totalTime": 22.501,
      "stopsDetail": [{"lap": 26, "time": 22.501}]
    },
    {
      "driver": "Alexander Albon",
      "stops": 1,
      "totalTime": 21.906,
      "stopsDetail": [{"lap": 26, "time": 21.906}]
    },
    {
      "driver": "Yuki Tsunoda",
      "stops": 1,
      "totalTime": 22.560,
      "stopsDetail": [{"lap": 27, "time": 22.560}]
    },
    {
      "driver": "Lewis Hamilton",
      "stops": 1,
      "totalTime": 23.104,
      "stopsDetail": [{"lap": 28, "time": 23.104}]
    },
    {
      "driver": "Fernando Alonso",
      "stops": 1,
      "totalTime": 23.602,
      "stopsDetail": [{"lap": 28, "time": 23.602}]
    },
    {
      "driver": "Liam Lawson",
      "stops": 1,
      "totalTime": 22.439,
      "stopsDetail": [{"lap": 28, "time": 22.439}]
    },
    {
      "driver": "Oscar Piastri",
      "stops": 1,
      "totalTime": 22.106,
      "stopsDetail": [{"lap": 29, "time": 22.106}]
    },
    {
      "driver": "Lando Norris",
      "stops": 1,
      "totalTime": 22.044,
      "stopsDetail": [{"lap": 29, "time": 22.044}]
    },
    {
      "driver": "George Russell",
      "stops": 1,
      "totalTime": 24.257,
      "stopsDetail": [{"lap": 29, "time": 24.257}]
    },
    {
      "driver": "Charles Leclerc",
      "stops": 1,
      "totalTime": 22.183,
      "stopsDetail": [{"lap": 29, "time": 22.183}]
    },
    {
      "driver": "Pierre Gasly",
      "stops": 1,
      "totalTime": 22.060,
      "stopsDetail": [{"lap": 32, "time": 22.060}]
    },
    {
      "driver": "Nico Hulkenberg",
      "stops": 1,
      "totalTime": 22.108,
      "stopsDetail": [{"lap": 36, "time": 22.108}]
    }
  ],
  "fastestLaps": [
    {"driver": "Lando Norris", "lapTime": "1:29.746", "lap": 36, "avgSpeed": "217.092 km/h"},
    {"driver": "Oscar Piastri", "lapTime": "1:29.822", "lap": 35, "avgSpeed": "216.908 km/h"},
    {"driver": "George Russell", "lapTime": "1:30.318", "lap": 31, "avgSpeed": "215.717 km/h"},
    {"driver": "Charles Leclerc", "lapTime": "1:30.461", "lap": 35, "avgSpeed": "215.376 km/h"},
    {"driver": "Max Verstappen", "lapTime": "1:30.466", "lap": 41, "avgSpeed": "215.364 km/h"},
    {"driver": "Alexander Albon", "lapTime": "1:30.482", "lap": 55, "avgSpeed": "215.326 km/h"},
    {"driver": "Lewis Hamilton", "lapTime": "1:30.562", "lap": 35, "avgSpeed": "215.136 km/h"},
    {"driver": "Carlos Sainz", "lapTime": "1:30.703", "lap": 35, "avgSpeed": "214.802 km/h"},
    {"driver": "Kimi Antonelli", "lapTime": "1:30.795", "lap": 27, "avgSpeed": "214.584 km/h"},
    {"driver": "Yuki Tsunoda", "lapTime": "1:30.964", "lap": 55, "avgSpeed": "214.185 km/h"},
    {"driver": "Isack Hadjar", "lapTime": "1:30.971", "lap": 51, "avgSpeed": "214.169 km/h"},
    {"driver": "Nico Hulkenberg", "lapTime": "1:31.015", "lap": 43, "avgSpeed": "214.065 km/h"},
    {"driver": "Esteban Ocon", "lapTime": "1:31.122", "lap": 30, "avgSpeed": "213.814 km/h"},
    {"driver": "Pierre Gasly", "lapTime": "1:31.159", "lap": 35, "avgSpeed": "213.727 km/h"},
    {"driver": "Fernando Alonso", "lapTime": "1:31.287", "lap": 38, "avgSpeed": "213.427 km/h"},
    {"driver": "Lance Stroll", "lapTime": "1:31.769", "lap": 51, "avgSpeed": "212.304 km/h"},
    {"driver": "Gabriel Bortoleto", "lapTime": "1:32.328", "lap": 21, "avgSpeed": "211.021 km/h"},
    {"driver": "Oliver Bearman", "lapTime": "1:32.680", "lap": 24, "avgSpeed": "210.220 km/h"}
  ]
},
"jeddah": {
  "round": 5,
  "name": "Saudi Arabian Grand Prix",
  "date": "Apr 18-20, 2025",
  "track": "Jeddah Corniche Circuit",
  "trackLength": 6.174,
  "weather": "Dry",
  "safetyCars": 0,
  "winner": {
    "driver": "Oscar Piastri",
    "team": "McLaren"
  },
  "fastestLap": {
    "driver": "Lando Norris",
    "time": "1:31.778",
    "lap": 41
  },
  "results": {
    "qualifying": [
      {"pos": 1, "driver": "Max Verstappen", "team": "Red Bull", "q1": "1:27.778", "q2": "1:27.529", "q3": "1:27.294"},
      {"pos": 2, "driver": "Oscar Piastri", "team": "McLaren", "q1": "1:27.901", "q2": "1:27.545", "q3": "1:27.304"},
      {"pos": 3, "driver": "George Russell", "team": "Mercedes", "q1": "1:28.282", "q2": "1:27.599", "q3": "1:27.407"},
      {"pos": 4, "driver": "Charles Leclerc", "team": "Ferrari", "q1": "1:28.552", "q2": "1:27.866", "q3": "1:27.670"},
      {"pos": 5, "driver": "Kimi Antonelli", "team": "Mercedes", "q1": "1:28.128", "q2": "1:27.798", "q3": "1:27.866"},
      {"pos": 6, "driver": "Carlos Sainz", "team": "Williams", "q1": "1:28.354", "q2": "1:28.024", "q3": "1:28.164"},
      {"pos": 7, "driver": "Lewis Hamilton", "team": "Ferrari", "q1": "1:28.372", "q2": "1:28.102", "q3": "1:28.201"},
      {"pos": 8, "driver": "Yuki Tsunoda", "team": "Red Bull", "q1": "1:28.226", "q2": "1:27.990", "q3": "1:28.204"},
      {"pos": 9, "driver": "Pierre Gasly", "team": "Alpine", "q1": "1:28.421", "q2": "1:28.025", "q3": "1:28.367"},
      {"pos": 10, "driver": "Lando Norris", "team": "McLaren", "q1": "1:27.805", "q2": "1:27.481", "q3": null},
      {"pos": 11, "driver": "Alexander Albon", "team": "Williams", "q1": "1:28.279", "q2": "1:28.109", "q3": null},
      {"pos": 12, "driver": "Liam Lawson", "team": "RB", "q1": "1:28.561", "q2": "1:28.191", "q3": null},
      {"pos": 13, "driver": "Fernando Alonso", "team": "Aston Martin", "q1": "1:28.548", "q2": "1:28.303", "q3": null},
      {"pos": 14, "driver": "Isack Hadjar", "team": "RB", "q1": "1:28.571", "q2": "1:28.418", "q3": null},
      {"pos": 15, "driver": "Oliver Bearman", "team": "Haas", "q1": "1:28.536", "q2": "1:28.648", "q3": null},
      {"pos": 16, "driver": "Lance Stroll", "team": "Aston Martin", "q1": "1:28.645", "q2": null, "q3": null},
      {"pos": 17, "driver": "Jack Doohan", "team": "Alpine", "q1": "1:28.739", "q2": null, "q3": null},
      {"pos": 18, "driver": "Nico Hulkenberg", "team": "Sauber", "q1": "1:28.782", "q2": null, "q3": null},
      {"pos": 19, "driver": "Esteban Ocon", "team": "Haas", "q1": "1:29.092", "q2": null, "q3": null},
      {"pos": 20, "driver": "Gabriel Bortoleto", "team": "Sauber", "q1": "1:29.462", "q2": null, "q3": null}
    ],
    "race": [
      {"pos": 1, "driver": "Oscar Piastri", "team": "McLaren", "start": 2, "laps": 50, "points": 25},
      {"pos": 2, "driver": "Max Verstappen", "team": "Red Bull", "start": 1, "laps": 50, "points": 18},
      {"pos": 3, "driver": "Charles Leclerc", "team": "Ferrari", "start": 4, "laps": 50, "points": 15},
      {"pos": 4, "driver": "Lando Norris", "team": "McLaren", "start": 10, "laps": 50, "points": 12},
      {"pos": 5, "driver": "George Russell", "team": "Mercedes", "start": 3, "laps": 50, "points": 10},
      {"pos": 6, "driver": "Kimi Antonelli", "team": "Mercedes", "start": 5, "laps": 50, "points": 8},
      {"pos": 7, "driver": "Lewis Hamilton", "team": "Ferrari", "start": 7, "laps": 50, "points": 6},
      {"pos": 8, "driver": "Carlos Sainz", "team": "Williams", "start": 6, "laps": 50, "points": 4},
      {"pos": 9, "driver": "Alexander Albon", "team": "Williams", "start": 11, "laps": 50, "points": 2},
      {"pos": 10, "driver": "Isack Hadjar", "team": "RB", "start": 14, "laps": 50, "points": 1},
      {"pos": 11, "driver": "Fernando Alonso", "team": "Aston Martin", "start": 13, "laps": 50, "points": 0},
      {"pos": 12, "driver": "Liam Lawson", "team": "RB", "start": 12, "laps": 50, "points": 0},
      {"pos": 13, "driver": "Oliver Bearman", "team": "Haas", "start": 15, "laps": 50, "points": 0},
      {"pos": 14, "driver": "Esteban Ocon", "team": "Haas", "start": 19, "laps": 50, "points": 0},
      {"pos": 15, "driver": "Nico Hulkenberg", "team": "Sauber", "start": 18, "laps": 49, "points": 0},
      {"pos": 16, "driver": "Lance Stroll", "team": "Aston Martin", "start": 16, "laps": 49, "points": 0},
      {"pos": 17, "driver": "Jack Doohan", "team": "Alpine", "start": 17, "laps": 49, "points": 0},
      {"pos": 18, "driver": "Gabriel Bortoleto", "team": "Sauber", "start": 20, "laps": 49, "points": 0},
      {"pos": 19, "driver": "Yuki Tsunoda", "team": "Red Bull", "start": 8, "laps": 1, "points": 0},
      {"pos": 20, "driver": "Pierre Gasly", "team": "Alpine", "start": 9, "laps": 0, "points": 0}
    ]
  },
  "highlights": [
    "Piastri wins thrilling battle with Verstappen",
    "McLaren double podium with Norris P4",
    "Norris recovers from P10 start to P4",
    "Mercedes duo Russell and Antonelli in top 6",
    "Early retirements for Tsunoda and Gasly"
  ],
  "practice": {
    "p1": [
      {"driver": "Pierre Gasly", "team": "Alpine", "time": "1:29.239", "laps": 25},
      {"driver": "Lando Norris", "team": "McLaren", "time": "1:29.246", "laps": 25},
      {"driver": "Charles Leclerc", "team": "Ferrari", "time": "1:29.309", "laps": 28},
      {"driver": "Oscar Piastri", "team": "McLaren", "time": "1:29.341", "laps": 25},
      {"driver": "Alexander Albon", "team": "Williams", "time": "1:29.606", "laps": 23},
      {"driver": "George Russell", "team": "Mercedes", "time": "1:29.618", "laps": 21},
      {"driver": "Carlos Sainz", "team": "Williams", "time": "1:29.779", "laps": 26},
      {"driver": "Lewis Hamilton", "team": "Ferrari", "time": "1:29.815", "laps": 27},
      {"driver": "Max Verstappen", "team": "Red Bull", "time": "1:29.818", "laps": 26},
      {"driver": "Yuki Tsunoda", "team": "Red Bull", "time": "1:29.821", "laps": 26},
      {"driver": "Liam Lawson", "team": "RB", "time": "1:29.907", "laps": 19},
      {"driver": "Nico Hulkenberg", "team": "Sauber", "time": "1:29.916", "laps": 25},
      {"driver": "Kimi Antonelli", "team": "Mercedes", "time": "1:29.934", "laps": 24},
      {"driver": "Fernando Alonso", "team": "Aston Martin", "time": "1:29.976", "laps": 24},
      {"driver": "Isack Hadjar", "team": "RB", "time": "1:30.011", "laps": 15},
      {"driver": "Jack Doohan", "team": "Alpine", "time": "1:30.183", "laps": 25},
      {"driver": "Lance Stroll", "team": "Aston Martin", "time": "1:30.583", "laps": 23},
      {"driver": "Oliver Bearman", "team": "Haas", "time": "1:30.595", "laps": 21},
      {"driver": "Esteban Ocon", "team": "Haas", "time": "1:31.029", "laps": 21},
      {"driver": "Gabriel Bortoleto", "team": "Sauber", "time": "1:31.038", "laps": 23}
    ],
    "p2": [
      {"driver": "Lando Norris", "team": "McLaren", "time": "1:28.267", "laps": 21},
      {"driver": "Oscar Piastri", "team": "McLaren", "time": "1:28.430", "laps": 22},
      {"driver": "Max Verstappen", "team": "Red Bull", "time": "1:28.547", "laps": 23},
      {"driver": "Charles Leclerc", "team": "Ferrari", "time": "1:28.749", "laps": 22},
      {"driver": "Carlos Sainz", "team": "Williams", "time": "1:28.942", "laps": 24},
      {"driver": "Yuki Tsunoda", "team": "Red Bull", "time": "1:28.963", "laps": 19},
      {"driver": "George Russell", "team": "Mercedes", "time": "1:28.973", "laps": 21},
      {"driver": "Pierre Gasly", "team": "Alpine", "time": "1:29.106", "laps": 22},
      {"driver": "Nico Hulkenberg", "team": "Sauber", "time": "1:29.193", "laps": 21},
      {"driver": "Alexander Albon", "team": "Williams", "time": "1:29.220", "laps": 23},
      {"driver": "Kimi Antonelli", "team": "Mercedes", "time": "1:29.242", "laps": 16},
      {"driver": "Isack Hadjar", "team": "RB", "time": "1:29.306", "laps": 17},
      {"driver": "Lewis Hamilton", "team": "Ferrari", "time": "1:29.371", "laps": 23},
      {"driver": "Liam Lawson", "team": "RB", "time": "1:29.488", "laps": 22},
      {"driver": "Fernando Alonso", "team": "Aston Martin", "time": "1:29.662", "laps": 18},
      {"driver": "Oliver Bearman", "team": "Haas", "time": "1:29.754", "laps": 19},
      {"driver": "Jack Doohan", "team": "Alpine", "time": "1:29.912", "laps": 21},
      {"driver": "Lance Stroll", "team": "Aston Martin", "time": "1:30.007", "laps": 18},
      {"driver": "Esteban Ocon", "team": "Haas", "time": "1:30.019", "laps": 22}
    ],
    "p3": [
      {"driver": "Lando Norris", "team": "McLaren", "time": "1:27.489", "laps": 18},
      {"driver": "Oscar Piastri", "team": "McLaren", "time": "1:27.513", "laps": 19},
      {"driver": "George Russell", "team": "Mercedes", "time": "1:28.116", "laps": 17},
      {"driver": "Max Verstappen", "team": "Red Bull", "time": "1:28.334", "laps": 14},
      {"driver": "Charles Leclerc", "team": "Ferrari", "time": "1:28.372", "laps": 21},
      {"driver": "Alexander Albon", "team": "Williams", "time": "1:28.389", "laps": 16},
      {"driver": "Carlos Sainz", "team": "Williams", "time": "1:28.570", "laps": 15},
      {"driver": "Pierre Gasly", "team": "Alpine", "time": "1:28.625", "laps": 16},
      {"driver": "Yuki Tsunoda", "team": "Red Bull", "time": "1:28.670", "laps": 14},
      {"driver": "Kimi Antonelli", "team": "Mercedes", "time": "1:28.679", "laps": 18},
      {"driver": "Isack Hadjar", "team": "RB", "time": "1:28.769", "laps": 17},
      {"driver": "Lewis Hamilton", "team": "Ferrari", "time": "1:28.780", "laps": 20},
      {"driver": "Liam Lawson", "team": "RB", "time": "1:28.861", "laps": 18},
      {"driver": "Fernando Alonso", "team": "Aston Martin", "time": "1:28.888", "laps": 19},
      {"driver": "Jack Doohan", "team": "Alpine", "time": "1:28.898", "laps": 21},
      {"driver": "Oliver Bearman", "team": "Haas", "time": "1:28.989", "laps": 15},
      {"driver": "Nico Hulkenberg", "team": "Sauber", "time": "1:29.220", "laps": 16},
      {"driver": "Esteban Ocon", "team": "Haas", "time": "1:29.336", "laps": 20},
      {"driver": "Gabriel Bortoleto", "team": "Sauber", "time": "1:29.410", "laps": 24},
      {"driver": "Lance Stroll", "team": "Aston Martin", "time": "1:29.478", "laps": 19}
    ]
  },
  "pitStops": [
    {
      "driver": "Jack Doohan",
      "stops": 2,
      "totalTime": 42.092,
      "stopsDetail": [
        {"lap": 1, "time": 20.571},
        {"lap": 32, "time": 21.521}
      ]
    },
    {
      "driver": "Esteban Ocon",
      "stops": 1,
      "totalTime": 21.512,
      "stopsDetail": [{"lap": 1, "time": 21.512}]
    },
    {
      "driver": "Gabriel Bortoleto",
      "stops": 1,
      "totalTime": 21.854,
      "stopsDetail": [{"lap": 1, "time": 21.854}]
    },
    {
      "driver": "Oliver Bearman",
      "stops": 1,
      "totalTime": 22.095,
      "stopsDetail": [{"lap": 18, "time": 22.095}]
    },
    {
      "driver": "Oscar Piastri",
      "stops": 1,
      "totalTime": 21.388,
      "stopsDetail": [{"lap": 19, "time": 21.388}]
    },
    {
      "driver": "Kimi Antonelli",
      "stops": 1,
      "totalTime": 20.399,
      "stopsDetail": [{"lap": 19, "time": 20.399}]
    },
    {
      "driver": "Fernando Alonso",
      "stops": 1,
      "totalTime": 21.891,
      "stopsDetail": [{"lap": 19, "time": 21.891}]
    },
    {
      "driver": "George Russell",
      "stops": 1,
      "totalTime": 20.329,
      "stopsDetail": [{"lap": 20, "time": 20.329}]
    },
    {
      "driver": "Liam Lawson",
      "stops": 1,
      "totalTime": 20.534,
      "stopsDetail": [{"lap": 20, "time": 20.534}]
    },
    {
      "driver": "Max Verstappen",
      "stops": 1,
      "totalTime": 26.030,
      "stopsDetail": [{"lap": 21, "time": 26.030}]
    },
    {
      "driver": "Carlos Sainz",
      "stops": 1,
      "totalTime": 24.148,
      "stopsDetail": [{"lap": 21, "time": 24.148}]
    },
    {
      "driver": "Alexander Albon",
      "stops": 1,
      "totalTime": 22.135,
      "stopsDetail": [{"lap": 22, "time": 22.135}]
    },
    {
      "driver": "Lewis Hamilton",
      "stops": 1,
      "totalTime": 20.144,
      "stopsDetail": [{"lap": 23, "time": 20.144}]
    },
    {
      "driver": "Charles Leclerc",
      "stops": 1,
      "totalTime": 20.131,
      "stopsDetail": [{"lap": 29, "time": 20.131}]
    },
    {
      "driver": "Nico Hulkenberg",
      "stops": 1,
      "totalTime": 20.441,
      "stopsDetail": [{"lap": 33, "time": 20.441}]
    },
    {
      "driver": "Lando Norris",
      "stops": 1,
      "totalTime": 20.427,
      "stopsDetail": [{"lap": 34, "time": 20.427}]
    },
    {
      "driver": "Isack Hadjar",
      "stops": 1,
      "totalTime": 20.827,
      "stopsDetail": [{"lap": 34, "time": 20.827}]
    },
    {
      "driver": "Lance Stroll",
      "stops": 1,
      "totalTime": 20.487,
      "stopsDetail": [{"lap": 39, "time": 20.487}]
    }
  ],
  "fastestLaps": [
    {"driver": "Lando Norris", "lapTime": "1:31.778", "lap": 41, "avgSpeed": "242.175 km/h"},
    {"driver": "Charles Leclerc", "lapTime": "1:32.192", "lap": 49, "avgSpeed": "241.088 km/h"},
    {"driver": "Oscar Piastri", "lapTime": "1:32.228", "lap": 50, "avgSpeed": "240.994 km/h"},
    {"driver": "Max Verstappen", "lapTime": "1:32.280", "lap": 49, "avgSpeed": "240.858 km/h"},
    {"driver": "Kimi Antonelli", "lapTime": "1:32.396", "lap": 50, "avgSpeed": "240.555 km/h"},
    {"driver": "Carlos Sainz", "lapTime": "1:32.466", "lap": 50, "avgSpeed": "240.373 km/h"},
    {"driver": "Lewis Hamilton", "lapTime": "1:32.600", "lap": 43, "avgSpeed": "240.025 km/h"},
    {"driver": "Lance Stroll", "lapTime": "1:32.745", "lap": 44, "avgSpeed": "239.650 km/h"},
    {"driver": "George Russell", "lapTime": "1:32.893", "lap": 32, "avgSpeed": "239.268 km/h"},
    {"driver": "Liam Lawson", "lapTime": "1:32.998", "lap": 43, "avgSpeed": "238.998 km/h"},
    {"driver": "Fernando Alonso", "lapTime": "1:33.009", "lap": 49, "avgSpeed": "238.970 km/h"},
    {"driver": "Jack Doohan", "lapTime": "1:33.150", "lap": 49, "avgSpeed": "238.608 km/h"},
    {"driver": "Oliver Bearman", "lapTime": "1:33.238", "lap": 50, "avgSpeed": "238.383 km/h"},
    {"driver": "Isack Hadjar", "lapTime": "1:33.257", "lap": 39, "avgSpeed": "238.334 km/h"},
    {"driver": "Nico Hulkenberg", "lapTime": "1:33.446", "lap": 39, "avgSpeed": "237.852 km/h"},
    {"driver": "Alexander Albon", "lapTime": "1:33.477", "lap": 47, "avgSpeed": "237.773 km/h"},
    {"driver": "Esteban Ocon", "lapTime": "1:34.309", "lap": 47, "avgSpeed": "235.676 km/h"},
    {"driver": "Gabriel Bortoleto", "lapTime": "1:34.447", "lap": 39, "avgSpeed": "235.331 km/h"}
  ]
},
"bahrain": {
  "round": 4,
  "name": "Bahrain Grand Prix",
  "date": "Apr 11-13, 2025",
  "track": "Bahrain International Circuit",
  "trackLength": 5.412,
  "weather": "Dry",
  "safetyCars": 0,
  "winner": {
    "driver": "Oscar Piastri",
    "team": "McLaren"
  },
  "fastestLap": {
    "driver": "Oscar Piastri",
    "time": "1:35.140",
    "lap": 36
  },
  "results": {
    "qualifying": [
      {"pos": 1, "driver": "Oscar Piastri", "team": "McLaren", "q1": "1:31.392", "q2": "1:30.454", "q3": "1:29.841"},
      {"pos": 2, "driver": "George Russell", "team": "Mercedes", "q1": "1:31.494", "q2": "1:30.664", "q3": "1:30.009"},
      {"pos": 3, "driver": "Charles Leclerc", "team": "Ferrari", "q1": "1:31.454", "q2": "1:30.724", "q3": "1:30.175"},
      {"pos": 4, "driver": "Kimi Antonelli", "team": "Mercedes", "q1": "1:31.415", "q2": "1:30.716", "q3": "1:30.213"},
      {"pos": 5, "driver": "Pierre Gasly", "team": "Alpine", "q1": "1:31.462", "q2": "1:30.643", "q3": "1:30.216"},
      {"pos": 6, "driver": "Lando Norris", "team": "McLaren", "q1": "1:31.107", "q2": "1:30.560", "q3": "1:30.267"},
      {"pos": 7, "driver": "Max Verstappen", "team": "Red Bull", "q1": "1:31.303", "q2": "1:31.019", "q3": "1:30.423"},
      {"pos": 8, "driver": "Carlos Sainz", "team": "Williams", "q1": "1:31.591", "q2": "1:30.844", "q3": "1:30.680"},
      {"pos": 9, "driver": "Lewis Hamilton", "team": "Ferrari", "q1": "1:31.219", "q2": "1:31.009", "q3": "1:30.772"},
      {"pos": 10, "driver": "Yuki Tsunoda", "team": "Red Bull", "q1": "1:31.751", "q2": "1:31.228", "q3": "1:31.303"},
      {"pos": 11, "driver": "Jack Doohan", "team": "Alpine", "q1": "1:31.414", "q2": "1:31.245", "q3": null},
      {"pos": 12, "driver": "Isack Hadjar", "team": "RB", "q1": "1:31.591", "q2": "1:31.271", "q3": null},
      {"pos": 13, "driver": "Fernando Alonso", "team": "Aston Martin", "q1": "1:31.634", "q2": "1:31.886", "q3": null},
      {"pos": 14, "driver": "Esteban Ocon", "team": "Haas", "q1": "1:31.594", "q2": null, "q3": null},
      {"pos": 15, "driver": "Alexander Albon", "team": "Williams", "q1": "1:32.040", "q2": null, "q3": null},
      {"pos": 16, "driver": "Nico Hulkenberg", "team": "Sauber", "q1": "1:32.067", "q2": null, "q3": null},
      {"pos": 17, "driver": "Liam Lawson", "team": "RB", "q1": "1:32.165", "q2": null, "q3": null},
      {"pos": 18, "driver": "Gabriel Bortoleto", "team": "Sauber", "q1": "1:32.186", "q2": null, "q3": null},
      {"pos": 19, "driver": "Lance Stroll", "team": "Aston Martin", "q1": "1:32.283", "q2": null, "q3": null},
      {"pos": 20, "driver": "Oliver Bearman", "team": "Haas", "q1": "1:32.373", "q2": null, "q3": null}
    ],
    "race": [
      {"pos": 1, "driver": "Oscar Piastri", "team": "McLaren", "start": 1, "laps": 57, "points": 25},
      {"pos": 2, "driver": "George Russell", "team": "Mercedes", "start": 3, "laps": 57, "points": 18},
      {"pos": 3, "driver": "Lando Norris", "team": "McLaren", "start": 6, "laps": 57, "points": 15},
      {"pos": 4, "driver": "Charles Leclerc", "team": "Ferrari", "start": 2, "laps": 57, "points": 12},
      {"pos": 5, "driver": "Lewis Hamilton", "team": "Ferrari", "start": 9, "laps": 57, "points": 10},
      {"pos": 6, "driver": "Max Verstappen", "team": "Red Bull", "start": 7, "laps": 57, "points": 8},
      {"pos": 7, "driver": "Pierre Gasly", "team": "Alpine", "start": 4, "laps": 57, "points": 6},
      {"pos": 8, "driver": "Esteban Ocon", "team": "Haas", "start": 14, "laps": 57, "points": 4},
      {"pos": 9, "driver": "Yuki Tsunoda", "team": "Red Bull", "start": 10, "laps": 57, "points": 2},
      {"pos": 10, "driver": "Oliver Bearman", "team": "Haas", "start": 20, "laps": 57, "points": 1},
      {"pos": 11, "driver": "Kimi Antonelli", "team": "Mercedes", "start": 5, "laps": 57, "points": 0},
      {"pos": 12, "driver": "Alexander Albon", "team": "Williams", "start": 15, "laps": 57, "points": 0},
      {"pos": 13, "driver": "Isack Hadjar", "team": "RB", "start": 12, "laps": 57, "points": 0},
      {"pos": 14, "driver": "Jack Doohan", "team": "Alpine", "start": 11, "laps": 57, "points": 0},
      {"pos": 15, "driver": "Fernando Alonso", "team": "Aston Martin", "start": 13, "laps": 57, "points": 0},
      {"pos": 16, "driver": "Liam Lawson", "team": "RB", "start": 17, "laps": 57, "points": 0},
      {"pos": 17, "driver": "Lance Stroll", "team": "Aston Martin", "start": 19, "laps": 57, "points": 0},
      {"pos": 18, "driver": "Gabriel Bortoleto", "team": "Sauber", "start": 18, "laps": 57, "points": 0},
      {"pos": 19, "driver": "Carlos Sainz", "team": "Williams", "start": 8, "laps": 45, "points": 0},
      {"pos": 20, "driver": "Nico Hulkenberg", "team": "Sauber", "start": 16, "laps": 0, "points": 0}
    ]
  },
  "highlights": [
    "Piastri dominates Bahrain GP for victory",
    "McLaren double podium with Norris P3",
    "Russell holds off Leclerc for P2",
    "Bearman scores point on debut from P20",
    "Hulkenberg disqualified post-race"
  ],
  "practice": {
    "p1": [
      {"driver": "Lando Norris", "team": "McLaren", "time": "1:33.204", "laps": 23},
      {"driver": "Pierre Gasly", "team": "Alpine", "time": "1:33.442", "laps": 23},
      {"driver": "Lewis Hamilton", "team": "Ferrari", "time": "1:33.800", "laps": 23},
      {"driver": "Alexander Albon", "team": "Williams", "time": "1:33.928", "laps": 24},
      {"driver": "Esteban Ocon", "team": "Haas", "time": "1:34.184", "laps": 19},
      {"driver": "Nico Hulkenberg", "team": "Sauber", "time": "1:34.262", "laps": 24},
      {"driver": "Jack Doohan", "team": "Alpine", "time": "1:34.396", "laps": 23},
      {"driver": "Liam Lawson", "team": "RB", "time": "1:34.397", "laps": 23},
      {"driver": "Yuki Tsunoda", "team": "Red Bull", "time": "1:34.484", "laps": 23},
      {"driver": "Oscar Piastri", "team": "McLaren", "time": "1:34.508", "laps": 25},
      {"driver": "Gabriel Bortoleto", "team": "Sauber", "time": "1:34.628", "laps": 23},
      {"driver": "Isack Hadjar", "team": "RB", "time": "1:34.667", "laps": 22},
      {"driver": "Luke Browning", "team": "Williams", "time": "1:34.885", "laps": 20},
      {"driver": "Dino Beganovic", "team": "Ferrari", "time": "1:35.055", "laps": 20},
      {"driver": "Lance Stroll", "team": "Aston Martin", "time": "1:35.116", "laps": 23},
      {"driver": "Felipe Drugovich", "team": "Aston Martin", "time": "1:35.198", "laps": 19},
      {"driver": "Ryo Hirakawa", "team": "Haas", "time": "1:35.261", "laps": 20},
      {"driver": "Frederik Vesti", "team": "Mercedes", "time": "1:35.325", "laps": 26},
      {"driver": "Ayumu Iwasa", "team": "Red Bull", "time": "1:35.475", "laps": 20},
      {"driver": "Kimi Antonelli", "team": "Mercedes", "time": "1:38.051", "laps": 3}
    ],
    "p2": [
      {"driver": "Oscar Piastri", "team": "McLaren", "time": "1:30.505", "laps": 29},
      {"driver": "Lando Norris", "team": "McLaren", "time": "1:30.659", "laps": 28},
      {"driver": "George Russell", "team": "Mercedes", "time": "1:31.032", "laps": 25},
      {"driver": "Charles Leclerc", "team": "Ferrari", "time": "1:31.045", "laps": 27},
      {"driver": "Kimi Antonelli", "team": "Mercedes", "time": "1:31.227", "laps": 27},
      {"driver": "Isack Hadjar", "team": "RB", "time": "1:31.238", "laps": 24},
      {"driver": "Max Verstappen", "team": "Red Bull", "time": "1:31.330", "laps": 27},
      {"driver": "Lewis Hamilton", "team": "Ferrari", "time": "1:31.576", "laps": 23},
      {"driver": "Oliver Bearman", "team": "Haas", "time": "1:31.584", "laps": 27},
      {"driver": "Carlos Sainz", "team": "Williams", "time": "1:31.623", "laps": 28},
      {"driver": "Alexander Albon", "team": "Williams", "time": "1:31.696", "laps": 29},
      {"driver": "Liam Lawson", "team": "RB", "time": "1:31.706", "laps": 27},
      {"driver": "Gabriel Bortoleto", "team": "Sauber", "time": "1:31.772", "laps": 25},
      {"driver": "Jack Doohan", "team": "Alpine", "time": "1:31.788", "laps": 27},
      {"driver": "Fernando Alonso", "team": "Aston Martin", "time": "1:31.825", "laps": 19},
      {"driver": "Esteban Ocon", "team": "Haas", "time": "1:31.870", "laps": 27},
      {"driver": "Pierre Gasly", "team": "Alpine", "time": "1:31.947", "laps": 27},
      {"driver": "Yuki Tsunoda", "team": "Red Bull", "time": "1:32.024", "laps": 25},
      {"driver": "Lance Stroll", "team": "Aston Martin", "time": "1:32.382", "laps": 27},
      {"driver": "Nico Hulkenberg", "team": "Sauber", "time": "1:32.496", "laps": 24}
    ],
    "p3": [
      {"driver": "Oscar Piastri", "team": "McLaren", "time": "1:31.646", "laps": 13},
      {"driver": "Lando Norris", "team": "McLaren", "time": "1:32.314", "laps": 18},
      {"driver": "Charles Leclerc", "team": "Ferrari", "time": "1:32.480", "laps": 19},
      {"driver": "George Russell", "team": "Mercedes", "time": "1:32.827", "laps": 13},
      {"driver": "Kimi Antonelli", "team": "Mercedes", "time": "1:32.916", "laps": 12},
      {"driver": "Pierre Gasly", "team": "Alpine", "time": "1:32.974", "laps": 18},
      {"driver": "Isack Hadjar", "team": "RB", "time": "1:33.023", "laps": 15},
      {"driver": "Max Verstappen", "team": "Red Bull", "time": "1:33.027", "laps": 13},
      {"driver": "Carlos Sainz", "team": "Williams", "time": "1:33.092", "laps": 15},
      {"driver": "Lewis Hamilton", "team": "Ferrari", "time": "1:33.111", "laps": 18},
      {"driver": "Esteban Ocon", "team": "Haas", "time": "1:33.240", "laps": 17},
      {"driver": "Jack Doohan", "team": "Alpine", "time": "1:33.347", "laps": 18},
      {"driver": "Liam Lawson", "team": "RB", "time": "1:33.370", "laps": 14},
      {"driver": "Fernando Alonso", "team": "Aston Martin", "time": "1:33.548", "laps": 18},
      {"driver": "Alexander Albon", "team": "Williams", "time": "1:33.753", "laps": 16},
      {"driver": "Oliver Bearman", "team": "Haas", "time": "1:34.335", "laps": 18},
      {"driver": "Lance Stroll", "team": "Aston Martin", "time": "1:34.363", "laps": 16},
      {"driver": "Gabriel Bortoleto", "team": "Sauber", "time": "1:34.518", "laps": 16},
      {"driver": "Nico Hulkenberg", "team": "Sauber", "time": "1:34.636", "laps": 7},
      {"driver": "Yuki Tsunoda", "team": "Red Bull", "time": "1:34.965", "laps": 15}
    ]
  },
  "pitStops": [
    {
      "driver": "Nico Hulkenberg",
      "stops": 1,
      "totalTime": 24.229,
      "stopsDetail": [{"lap": 5, "time": 24.229}]
    },
    {
      "driver": "Isack Hadjar",
      "stops": 1,
      "totalTime": 25.265,
      "stopsDetail": [{"lap": 6, "time": 25.265}]
    },
    {
      "driver": "Esteban Ocon",
      "stops": 2,
      "totalTime": 49.708,
      "stopsDetail": [
        {"lap": 8, "time": 24.876},
        {"lap": 27, "time": 24.832}
      ]
    },
    {
      "driver": "Jack Doohan",
      "stops": 2,
      "totalTime": 49.884,
      "stopsDetail": [
        {"lap": 9, "time": 24.791},
        {"lap": 28, "time": 25.093}
      ]
    },
    {
      "driver": "Lando Norris",
      "stops": 2,
      "totalTime": 53.857,
      "stopsDetail": [
        {"lap": 10, "time": 29.822},
        {"lap": 32, "time": 24.035}
      ]
    },
    {
      "driver": "Pierre Gasly",
      "stops": 2,
      "totalTime": 49.991,
      "stopsDetail": [
        {"lap": 10, "time": 25.033},
        {"lap": 28, "time": 24.958}
      ]
    },
    {
      "driver": "Max Verstappen",
      "stops": 2,
      "totalTime": 54.585,
      "stopsDetail": [
        {"lap": 10, "time": 26.518},
        {"lap": 26, "time": 28.067}
      ]
    },
    {
      "driver": "Yuki Tsunoda",
      "stops": 2,
      "totalTime": 51.264,
      "stopsDetail": [
        {"lap": 11, "time": 26.493},
        {"lap": 32, "time": 24.771}
      ]
    },
    {
      "driver": "Kimi Antonelli",
      "stops": 3,
      "totalTime": 74.006,
      "stopsDetail": [
        {"lap": 12, "time": 24.807},
        {"lap": 27, "time": 24.506},
        {"lap": 32, "time": 24.693}
      ]
    },
    {
      "driver": "Lance Stroll",
      "stops": 2,
      "totalTime": 49.203,
      "stopsDetail": [
        {"lap": 12, "time": 24.611},
        {"lap": 32, "time": 24.592}
      ]
    },
    {
      "driver": "George Russell",
      "stops": 2,
      "totalTime": 48.144,
      "stopsDetail": [
        {"lap": 13, "time": 24.044},
        {"lap": 32, "time": 24.100}
      ]
    },
    {
      "driver": "Gabriel Bortoleto",
      "stops": 2,
      "totalTime": 49.539,
      "stopsDetail": [
        {"lap": 13, "time": 24.143},
        {"lap": 32, "time": 25.396}
      ]
    },
    {
      "driver": "Oscar Piastri",
      "stops": 2,
      "totalTime": 49.398,
      "stopsDetail": [
        {"lap": 14, "time": 24.498},
        {"lap": 32, "time": 24.900}
      ]
    },
    {
      "driver": "Carlos Sainz",
      "stops": 3,
      "totalTime": 86.031,
      "stopsDetail": [
        {"lap": 14, "time": 24.600},
        {"lap": 32, "time": 24.315},
        {"lap": 44, "time": 37.116}
      ]
    },
    {
      "driver": "Oliver Bearman",
      "stops": 2,
      "totalTime": 49.457,
      "stopsDetail": [
        {"lap": 14, "time": 24.562},
        {"lap": 32, "time": 24.895}
      ]
    },
    {
      "driver": "Liam Lawson",
      "stops": 2,
      "totalTime": 49.433,
      "stopsDetail": [
        {"lap": 14, "time": 25.242},
        {"lap": 32, "time": 24.191}
      ]
    },
    {
      "driver": "Alexander Albon",
      "stops": 2,
      "totalTime": 51.384,
      "stopsDetail": [
        {"lap": 16, "time": 25.394},
        {"lap": 32, "time": 25.990}
      ]
    },
    {
      "driver": "Fernando Alonso",
      "stops": 2,
      "totalTime": 49.672,
      "stopsDetail": [
        {"lap": 16, "time": 24.826},
        {"lap": 32, "time": 24.846}
      ]
    },
    {
      "driver": "Charles Leclerc",
      "stops": 2,
      "totalTime": 49.043,
      "stopsDetail": [
        {"lap": 17, "time": 24.802},
        {"lap": 32, "time": 24.241}
      ]
    },
    {
      "driver": "Lewis Hamilton",
      "stops": 2,
      "totalTime": 48.690,
      "stopsDetail": [
        {"lap": 17, "time": 24.340},
        {"lap": 32, "time": 24.350}
      ]
    }
  ],
  "fastestLaps": [
    {"driver": "Oscar Piastri", "lapTime": "1:35.140", "lap": 36, "avgSpeed": "204.784 km/h"},
    {"driver": "George Russell", "lapTime": "1:35.518", "lap": 36, "avgSpeed": "203.974 km/h"},
    {"driver": "Lando Norris", "lapTime": "1:35.728", "lap": 38, "avgSpeed": "203.526 km/h"},
    {"driver": "Charles Leclerc", "lapTime": "1:36.132", "lap": 36, "avgSpeed": "202.671 km/h"},
    {"driver": "Max Verstappen", "lapTime": "1:36.167", "lap": 29, "avgSpeed": "202.597 km/h"},
    {"driver": "Lewis Hamilton", "lapTime": "1:36.235", "lap": 37, "avgSpeed": "202.454 km/h"},
    {"driver": "Pierre Gasly", "lapTime": "1:36.531", "lap": 39, "avgSpeed": "201.833 km/h"},
    {"driver": "Jack Doohan", "lapTime": "1:36.682", "lap": 31, "avgSpeed": "201.518 km/h"},
    {"driver": "Kimi Antonelli", "lapTime": "1:36.690", "lap": 29, "avgSpeed": "201.501 km/h"},
    {"driver": "Isack Hadjar", "lapTime": "1:36.952", "lap": 30, "avgSpeed": "200.957 km/h"},
    {"driver": "Carlos Sainz", "lapTime": "1:36.954", "lap": 16, "avgSpeed": "200.953 km/h"},
    {"driver": "Esteban Ocon", "lapTime": "1:37.098", "lap": 30, "avgSpeed": "200.655 km/h"},
    {"driver": "Alexander Albon", "lapTime": "1:37.141", "lap": 47, "avgSpeed": "200.566 km/h"},
    {"driver": "Yuki Tsunoda", "lapTime": "1:37.225", "lap": 45, "avgSpeed": "200.392 km/h"},
    {"driver": "Oliver Bearman", "lapTime": "1:37.303", "lap": 40, "avgSpeed": "200.232 km/h"},
    {"driver": "Nico Hulkenberg", "lapTime": "1:37.338", "lap": 30, "avgSpeed": "200.160 km/h"},
    {"driver": "Liam Lawson", "lapTime": "1:37.380", "lap": 44, "avgSpeed": "200.073 km/h"},
    {"driver": "Fernando Alonso", "lapTime": "1:37.906", "lap": 38, "avgSpeed": "198.999 km/h"},
    {"driver": "Gabriel Bortoleto", "lapTime": "1:38.006", "lap": 38, "avgSpeed": "198.795 km/h"},
    {"driver": "Lance Stroll", "lapTime": "1:38.064", "lap": 38, "avgSpeed": "198.678 km/h"}
  ]
},
"suzuka": {
  "round": 3,
  "name": "Japanese Grand Prix",
  "date": "Apr 4-6, 2025",
  "track": "Suzuka Circuit",
  "trackLength": 5.807,
  "weather": "Dry",
  "safetyCars": 0,
  "winner": {
    "driver": "Max Verstappen",
    "team": "Red Bull"
  },
  "fastestLap": {
    "driver": "Kimi Antonelli",
    "time": "1:30.965",
    "lap": 50
  },
  "results": {
    "qualifying": [
      {"pos": 1, "driver": "Max Verstappen", "team": "Red Bull", "q1": "1:27.943", "q2": "1:27.502", "q3": "1:26.983"},
      {"pos": 2, "driver": "Lando Norris", "team": "McLaren", "q1": "1:27.845", "q2": "1:27.146", "q3": "1:26.995"},
      {"pos": 3, "driver": "Oscar Piastri", "team": "McLaren", "q1": "1:27.687", "q2": "1:27.507", "q3": "1:27.027"},
      {"pos": 4, "driver": "Charles Leclerc", "team": "Ferrari", "q1": "1:27.920", "q2": "1:27.555", "q3": "1:27.299"},
      {"pos": 5, "driver": "George Russell", "team": "Mercedes", "q1": "1:27.843", "q2": "1:27.400", "q3": "1:27.318"},
      {"pos": 6, "driver": "Kimi Antonelli", "team": "Mercedes", "q1": "1:27.968", "q2": "1:27.639", "q3": "1:27.555"},
      {"pos": 7, "driver": "Isack Hadjar", "team": "RB", "q1": "1:28.278", "q2": "1:27.775", "q3": "1:27.569"},
      {"pos": 8, "driver": "Lewis Hamilton", "team": "Ferrari", "q1": "1:27.942", "q2": "1:27.610", "q3": "1:27.610"},
      {"pos": 9, "driver": "Alexander Albon", "team": "Williams", "q1": "1:28.218", "q2": "1:27.783", "q3": "1:27.615"},
      {"pos": 10, "driver": "Oliver Bearman", "team": "Haas", "q1": "1:28.228", "q2": "1:27.711", "q3": "1:27.867"},
      {"pos": 11, "driver": "Pierre Gasly", "team": "Alpine", "q1": "1:28.186", "q2": "1:27.822", "q3": null},
      {"pos": 12, "driver": "Carlos Sainz", "team": "Williams", "q1": "1:28.209", "q2": "1:27.836", "q3": null},
      {"pos": 13, "driver": "Fernando Alonso", "team": "Aston Martin", "q1": "1:28.337", "q2": "1:27.897", "q3": null},
      {"pos": 14, "driver": "Liam Lawson", "team": "RB", "q1": "1:28.554", "q2": "1:27.906", "q3": null},
      {"pos": 15, "driver": "Yuki Tsunoda", "team": "Red Bull", "q1": "1:27.967", "q2": "1:28.000", "q3": null},
      {"pos": 16, "driver": "Nico Hulkenberg", "team": "Sauber", "q1": "1:28.570", "q2": null, "q3": null},
      {"pos": 17, "driver": "Gabriel Bortoleto", "team": "Sauber", "q1": "1:28.622", "q2": null, "q3": null},
      {"pos": 18, "driver": "Esteban Ocon", "team": "Haas", "q1": "1:28.696", "q2": null, "q3": null},
      {"pos": 19, "driver": "Jack Doohan", "team": "Alpine", "q1": "1:28.877", "q2": null, "q3": null},
      {"pos": 20, "driver": "Lance Stroll", "team": "Aston Martin", "q1": "1:29.271", "q2": null, "q3": null}
    ],
    "race": [
      {"pos": 1, "driver": "Max Verstappen", "team": "Red Bull", "start": 1, "laps": 53, "points": 25},
      {"pos": 2, "driver": "Lando Norris", "team": "McLaren", "start": 2, "laps": 53, "points": 18},
      {"pos": 3, "driver": "Oscar Piastri", "team": "McLaren", "start": 3, "laps": 53, "points": 15},
      {"pos": 4, "driver": "Charles Leclerc", "team": "Ferrari", "start": 4, "laps": 53, "points": 12},
      {"pos": 5, "driver": "George Russell", "team": "Mercedes", "start": 5, "laps": 53, "points": 10},
      {"pos": 6, "driver": "Kimi Antonelli", "team": "Mercedes", "start": 6, "laps": 53, "points": 8},
      {"pos": 7, "driver": "Lewis Hamilton", "team": "Ferrari", "start": 8, "laps": 53, "points": 6},
      {"pos": 8, "driver": "Isack Hadjar", "team": "RB", "start": 7, "laps": 53, "points": 4},
      {"pos": 9, "driver": "Alexander Albon", "team": "Williams", "start": 9, "laps": 53, "points": 2},
      {"pos": 10, "driver": "Oliver Bearman", "team": "Haas", "start": 10, "laps": 53, "points": 1},
      {"pos": 11, "driver": "Fernando Alonso", "team": "Aston Martin", "start": 12, "laps": 53, "points": 0},
      {"pos": 12, "driver": "Yuki Tsunoda", "team": "Red Bull", "start": 14, "laps": 53, "points": 0},
      {"pos": 13, "driver": "Pierre Gasly", "team": "Alpine", "start": 11, "laps": 53, "points": 0},
      {"pos": 14, "driver": "Carlos Sainz", "team": "Williams", "start": 15, "laps": 53, "points": 0},
      {"pos": 15, "driver": "Jack Doohan", "team": "Alpine", "start": 19, "laps": 53, "points": 0},
      {"pos": 16, "driver": "Nico Hulkenberg", "team": "Sauber", "start": 16, "laps": 53, "points": 0},
      {"pos": 17, "driver": "Liam Lawson", "team": "RB", "start": 13, "laps": 53, "points": 0},
      {"pos": 18, "driver": "Esteban Ocon", "team": "Haas", "start": 18, "laps": 53, "points": 0},
      {"pos": 19, "driver": "Gabriel Bortoleto", "team": "Sauber", "start": 17, "laps": 53, "points": 0},
      {"pos": 20, "driver": "Lance Stroll", "team": "Aston Martin", "start": 20, "laps": 52, "points": 0}
    ]
  },
  "highlights": [
    "Verstappen wins thrilling battle with McLaren duo",
    "Norris and Piastri complete podium for McLaren",
    "Antonelli scores fastest lap for Mercedes",
    "Bearman scores first F1 point for Haas",
    "Sainz recovers from penalty to finish P14"
  ],
  "practice": {
    "p1": [
      {"driver": "Lando Norris", "team": "McLaren", "time": "1:28.549", "laps": 24},
      {"driver": "George Russell", "team": "Mercedes", "time": "1:28.712", "laps": 29},
      {"driver": "Charles Leclerc", "team": "Ferrari", "time": "1:28.965", "laps": 27},
      {"driver": "Lewis Hamilton", "team": "Ferrari", "time": "1:29.051", "laps": 25},
      {"driver": "Max Verstappen", "team": "Red Bull", "time": "1:29.065", "laps": 23},
      {"driver": "Yuki Tsunoda", "team": "Red Bull", "time": "1:29.172", "laps": 25},
      {"driver": "Fernando Alonso", "team": "Aston Martin", "time": "1:29.222", "laps": 25},
      {"driver": "Isack Hadjar", "team": "RB", "time": "1:29.225", "laps": 28},
      {"driver": "Kimi Antonelli", "team": "Mercedes", "time": "1:29.284", "laps": 28},
      {"driver": "Carlos Sainz", "team": "Williams", "time": "1:29.333", "laps": 28},
      {"driver": "Alexander Albon", "team": "Williams", "time": "1:29.392", "laps": 25},
      {"driver": "Ryo Hirakawa", "team": "Alpine", "time": "1:29.394", "laps": 24},
      {"driver": "Liam Lawson", "team": "RB", "time": "1:29.536", "laps": 28},
      {"driver": "Pierre Gasly", "team": "Alpine", "time": "1:29.547", "laps": 23},
      {"driver": "Oscar Piastri", "team": "McLaren", "time": "1:29.708", "laps": 25},
      {"driver": "Lance Stroll", "team": "Aston Martin", "time": "1:29.758", "laps": 25},
      {"driver": "Nico Hulkenberg", "team": "Sauber", "time": "1:30.023", "laps": 21},
      {"driver": "Oliver Bearman", "team": "Haas", "time": "1:30.077", "laps": 17},
      {"driver": "Esteban Ocon", "team": "Haas", "time": "1:30.123", "laps": 21},
      {"driver": "Gabriel Bortoleto", "team": "Sauber", "time": "1:30.147", "laps": 26}
    ],
    "p2": [
      {"driver": "Oscar Piastri", "team": "McLaren", "time": "1:28.114", "laps": 13},
      {"driver": "Lando Norris", "team": "McLaren", "time": "1:28.163", "laps": 12},
      {"driver": "Isack Hadjar", "team": "RB", "time": "1:28.518", "laps": 12},
      {"driver": "Lewis Hamilton", "team": "Ferrari", "time": "1:28.544", "laps": 14},
      {"driver": "Liam Lawson", "team": "RB", "time": "1:28.559", "laps": 13},
      {"driver": "George Russell", "team": "Mercedes", "time": "1:28.567", "laps": 13},
      {"driver": "Charles Leclerc", "team": "Ferrari", "time": "1:28.586", "laps": 14},
      {"driver": "Max Verstappen", "team": "Red Bull", "time": "1:28.670", "laps": 9},
      {"driver": "Pierre Gasly", "team": "Alpine", "time": "1:28.757", "laps": 13},
      {"driver": "Carlos Sainz", "team": "Williams", "time": "1:28.832", "laps": 9},
      {"driver": "Alexander Albon", "team": "Williams", "time": "1:29.023", "laps": 11},
      {"driver": "Nico Hulkenberg", "team": "Sauber", "time": "1:29.062", "laps": 12},
      {"driver": "Gabriel Bortoleto", "team": "Sauber", "time": "1:29.335", "laps": 13},
      {"driver": "Esteban Ocon", "team": "Haas", "time": "1:29.507", "laps": 13},
      {"driver": "Oliver Bearman", "team": "Haas", "time": "1:29.654", "laps": 10},
      {"driver": "Kimi Antonelli", "team": "Mercedes", "time": "1:29.733", "laps": 13},
      {"driver": "Fernando Alonso", "team": "Aston Martin", "time": "1:29.978", "laps": 5},
      {"driver": "Yuki Tsunoda", "team": "Red Bull", "time": "1:30.625", "laps": 12},
      {"driver": "Lance Stroll", "team": "Aston Martin", "time": "1:30.845", "laps": 12},
      {"driver": "Jack Doohan", "team": "Alpine", "time": "1:32.659", "laps": 4}
    ],
    "p3": [
      {"driver": "Lando Norris", "team": "McLaren", "time": "1:27.965", "laps": 17},
      {"driver": "Oscar Piastri", "team": "McLaren", "time": "1:27.991", "laps": 17},
      {"driver": "George Russell", "team": "Mercedes", "time": "1:28.077", "laps": 15},
      {"driver": "Charles Leclerc", "team": "Ferrari", "time": "1:28.414", "laps": 13},
      {"driver": "Max Verstappen", "team": "Red Bull", "time": "1:28.497", "laps": 21},
      {"driver": "Lewis Hamilton", "team": "Ferrari", "time": "1:28.524", "laps": 13},
      {"driver": "Alexander Albon", "team": "Williams", "time": "1:28.554", "laps": 19},
      {"driver": "Pierre Gasly", "team": "Alpine", "time": "1:28.603", "laps": 15},
      {"driver": "Yuki Tsunoda", "team": "Red Bull", "time": "1:28.785", "laps": 15},
      {"driver": "Isack Hadjar", "team": "RB", "time": "1:28.786", "laps": 16},
      {"driver": "Carlos Sainz", "team": "Williams", "time": "1:28.846", "laps": 22},
      {"driver": "Liam Lawson", "team": "RB", "time": "1:29.104", "laps": 19},
      {"driver": "Kimi Antonelli", "team": "Mercedes", "time": "1:29.126", "laps": 20},
      {"driver": "Jack Doohan", "team": "Alpine", "time": "1:29.767", "laps": 15},
      {"driver": "Fernando Alonso", "team": "Aston Martin", "time": "1:29.772", "laps": 17},
      {"driver": "Oliver Bearman", "team": "Haas", "time": "1:30.084", "laps": 17},
      {"driver": "Gabriel Bortoleto", "team": "Sauber", "time": "1:30.134", "laps": 13},
      {"driver": "Esteban Ocon", "team": "Haas", "time": "1:30.183", "laps": 17},
      {"driver": "Lance Stroll", "team": "Aston Martin", "time": "1:30.267", "laps": 16},
      {"driver": "Nico Hulkenberg", "team": "Sauber", "time": "1:30.621", "laps": 17}
    ]
  },
  "pitStops": [
    {
      "driver": "Lance Stroll",
      "stops": 2,
      "totalTime": 47.332,
      "stopsDetail": [
        {"lap": 9, "time": 23.724},
        {"lap": 30, "time": 23.608}
      ]
    },
    {
      "driver": "Jack Doohan",
      "stops": 1,
      "totalTime": 23.382,
      "stopsDetail": [{"lap": 15, "time": 23.382}]
    },
    {
      "driver": "George Russell",
      "stops": 1,
      "totalTime": 23.184,
      "stopsDetail": [{"lap": 19, "time": 23.184}]
    },
    {
      "driver": "Oscar Piastri",
      "stops": 1,
      "totalTime": 23.037,
      "stopsDetail": [{"lap": 20, "time": 23.037}]
    },
    {
      "driver": "Max Verstappen",
      "stops": 1,
      "totalTime": 24.397,
      "stopsDetail": [{"lap": 21, "time": 24.397}]
    },
    {
      "driver": "Lando Norris",
      "stops": 1,
      "totalTime": 23.222,
      "stopsDetail": [{"lap": 21, "time": 23.222}]
    },
    {
      "driver": "Charles Leclerc",
      "stops": 1,
      "totalTime": 23.346,
      "stopsDetail": [{"lap": 21, "time": 23.346}]
    },
    {
      "driver": "Nico Hulkenberg",
      "stops": 1,
      "totalTime": 23.515,
      "stopsDetail": [{"lap": 22, "time": 23.515}]
    },
    {
      "driver": "Oliver Bearman",
      "stops": 1,
      "totalTime": 24.834,
      "stopsDetail": [{"lap": 23, "time": 24.834}]
    },
    {
      "driver": "Yuki Tsunoda",
      "stops": 1,
      "totalTime": 24.181,
      "stopsDetail": [{"lap": 23, "time": 24.181}]
    },
    {
      "driver": "Alexander Albon",
      "stops": 1,
      "totalTime": 23.093,
      "stopsDetail": [{"lap": 24, "time": 23.093}]
    },
    {
      "driver": "Fernando Alonso",
      "stops": 1,
      "totalTime": 25.708,
      "stopsDetail": [{"lap": 24, "time": 25.708}]
    },
    {
      "driver": "Pierre Gasly",
      "stops": 1,
      "totalTime": 26.041,
      "stopsDetail": [{"lap": 24, "time": 26.041}]
    },
    {
      "driver": "Isack Hadjar",
      "stops": 1,
      "totalTime": 23.333,
      "stopsDetail": [{"lap": 25, "time": 23.333}]
    },
    {
      "driver": "Lewis Hamilton",
      "stops": 1,
      "totalTime": 22.937,
      "stopsDetail": [{"lap": 30, "time": 22.937}]
    },
    {
      "driver": "Kimi Antonelli",
      "stops": 1,
      "totalTime": 24.380,
      "stopsDetail": [{"lap": 31, "time": 24.380}]
    },
    {
      "driver": "Gabriel Bortoleto",
      "stops": 1,
      "totalTime": 23.419,
      "stopsDetail": [{"lap": 31, "time": 23.419}]
    },
    {
      "driver": "Esteban Ocon",
      "stops": 1,
      "totalTime": 24.464,
      "stopsDetail": [{"lap": 32, "time": 24.464}]
    },
    {
      "driver": "Liam Lawson",
      "stops": 1,
      "totalTime": 23.662,
      "stopsDetail": [{"lap": 33, "time": 23.662}]
    },
    {
      "driver": "Carlos Sainz",
      "stops": 1,
      "totalTime": 23.337,
      "stopsDetail": [{"lap": 33, "time": 23.337}]
    }
  ],
  "fastestLaps": [
    {"driver": "Kimi Antonelli", "lapTime": "1:30.965", "lap": 50, "avgSpeed": "229.815 km/h"},
    {"driver": "Oscar Piastri", "lapTime": "1:31.039", "lap": 53, "avgSpeed": "229.629 km/h"},
    {"driver": "Max Verstappen", "lapTime": "1:31.041", "lap": 52, "avgSpeed": "229.624 km/h"},
    {"driver": "Carlos Sainz", "lapTime": "1:31.106", "lap": 36, "avgSpeed": "229.460 km/h"},
    {"driver": "Lando Norris", "lapTime": "1:31.116", "lap": 51, "avgSpeed": "229.435 km/h"},
    {"driver": "Alexander Albon", "lapTime": "1:31.125", "lap": 52, "avgSpeed": "229.412 km/h"},
    {"driver": "Isack Hadjar", "lapTime": "1:31.317", "lap": 52, "avgSpeed": "228.929 km/h"},
    {"driver": "George Russell", "lapTime": "1:31.357", "lap": 51, "avgSpeed": "228.829 km/h"},
    {"driver": "Lewis Hamilton", "lapTime": "1:31.406", "lap": 51, "avgSpeed": "228.707 km/h"},
    {"driver": "Charles Leclerc", "lapTime": "1:31.469", "lap": 47, "avgSpeed": "228.549 km/h"},
    {"driver": "Fernando Alonso", "lapTime": "1:31.770", "lap": 51, "avgSpeed": "227.799 km/h"},
    {"driver": "Pierre Gasly", "lapTime": "1:31.820", "lap": 52, "avgSpeed": "227.675 km/h"},
    {"driver": "Yuki Tsunoda", "lapTime": "1:31.871", "lap": 51, "avgSpeed": "227.549 km/h"},
    {"driver": "Esteban Ocon", "lapTime": "1:31.967", "lap": 48, "avgSpeed": "227.311 km/h"},
    {"driver": "Oliver Bearman", "lapTime": "1:32.006", "lap": 49, "avgSpeed": "227.215 km/h"},
    {"driver": "Gabriel Bortoleto", "lapTime": "1:32.034", "lap": 45, "avgSpeed": "227.146 km/h"},
    {"driver": "Liam Lawson", "lapTime": "1:32.043", "lap": 39, "avgSpeed": "227.124 km/h"},
    {"driver": "Lance Stroll", "lapTime": "1:32.052", "lap": 53, "avgSpeed": "227.102 km/h"},
    {"driver": "Nico Hulkenberg", "lapTime": "1:32.572", "lap": 31, "avgSpeed": "225.826 km/h"},
    {"driver": "Jack Doohan", "lapTime": "1:32.685", "lap": 47, "avgSpeed": "225.551 km/h"}
  ]
},
  "china": {
    "round": 2,
    "name": "Chinese Grand Prix",
    "date": "Mar 21-23, 2025",
    "track": "Shanghai International Circuit",
    "trackLength": 5.451,
    "weather": "Dry",
    "safetyCars": 0,
    "winner": {
      "driver": "Oscar Piastri",
      "team": "McLaren"
    },
    "fastestLap": {
      "driver": "Lando Norris",
      "time": "1:35.454",
      "lap": 53
    },
    "results": {
      "practice": [
        {"pos": 1, "driver": "Lando Norris", "team": "McLaren", "time": "1:31.504", "laps": 23},
        {"pos": 2, "driver": "Charles Leclerc", "team": "Ferrari", "time": "1:31.958", "laps": 21},
        {"pos": 3, "driver": "Oscar Piastri", "team": "McLaren", "time": "1:32.153", "laps": 24},
        {"pos": 4, "driver": "Lewis Hamilton", "team": "Ferrari", "time": "1:32.195", "laps": 22},
        {"pos": 5, "driver": "George Russell", "team": "Mercedes", "time": "1:32.377", "laps": 26},
        {"pos": 6, "driver": "Nico Hulkenberg", "team": "Kick Sauber", "time": "1:32.507", "laps": 21},
        {"pos": 7, "driver": "Alexander Albon", "team": "Williams", "time": "1:32.687", "laps": 24},
        {"pos": 8, "driver": "Fernando Alonso", "team": "Aston Martin", "time": "1:32.766", "laps": 23},
        {"pos": 9, "driver": "Kimi Antonelli", "team": "Mercedes", "time": "1:32.874", "laps": 27},
        {"pos": 10, "driver": "Yuki Tsunoda", "team": "Racing Bulls", "time": "1:32.934", "laps": 23},
        {"pos": 11, "driver": "Oliver Bearman", "team": "Haas", "time": "1:32.967", "laps": 23},
        {"pos": 12, "driver": "Lance Stroll", "team": "Aston Martin", "time": "1:32.984", "laps": 19},
        {"pos": 13, "driver": "Esteban Ocon", "team": "Haas", "time": "1:33.056", "laps": 23},
        {"pos": 14, "driver": "Pierre Gasly", "team": "Alpine", "time": "1:33.123", "laps": 23},
        {"pos": 15, "driver": "Carlos Sainz", "team": "Williams", "time": "1:33.145", "laps": 26},
        {"pos": 16, "driver": "Max Verstappen", "team": "Red Bull", "time": "1:33.284", "laps": 24},
        {"pos": 17, "driver": "Isack Hadjar", "team": "Racing Bulls", "time": "1:33.385", "laps": 23},
        {"pos": 18, "driver": "Liam Lawson", "team": "Red Bull", "time": "1:33.631", "laps": 23},
        {"pos": 19, "driver": "Gabriel Bortoleto", "team": "Kick Sauber", "time": "1:33.822", "laps": 24},
        {"pos": 20, "driver": "Jack Doohan", "team": "Alpine", "time": "1:33.923", "laps": 17}
      ],
      "sprintQualifying": [
        {"pos": 1, "driver": "Lewis Hamilton", "team": "Ferrari", "q1": "1:31.212", "q2": "1:31.384", "q3": "1:30.849"},
        {"pos": 2, "driver": "Max Verstappen", "team": "Red Bull", "q1": "1:31.916", "q2": "1:31.521", "q3": "1:30.867"},
        {"pos": 3, "driver": "Oscar Piastri", "team": "McLaren", "q1": "1:31.723", "q2": "1:31.362", "q3": "1:30.929"},
        {"pos": 4, "driver": "Charles Leclerc", "team": "Ferrari", "q1": "1:31.518", "q2": "1:31.561", "q3": "1:31.057"},
        {"pos": 5, "driver": "George Russell", "team": "Mercedes", "q1": "1:31.952", "q2": "1:31.346", "q3": "1:31.169"},
        {"pos": 6, "driver": "Lando Norris", "team": "McLaren", "q1": "1:31.396", "q2": "1:31.174", "q3": "1:31.393"},
        {"pos": 7, "driver": "Kimi Antonelli", "team": "Mercedes", "q1": "1:31.999", "q2": "1:31.475", "q3": "1:31.738"},
        {"pos": 8, "driver": "Yuki Tsunoda", "team": "Racing Bulls", "q1": "1:32.316", "q2": "1:31.794", "q3": "1:31.773"},
        {"pos": 9, "driver": "Alexander Albon", "team": "Williams", "q1": "1:32.462", "q2": "1:31.539", "q3": "1:31.852"},
        {"pos": 10, "driver": "Lance Stroll", "team": "Aston Martin", "q1": "1:32.327", "q2": "1:31.742", "q3": "1:31.982"},
        {"pos": 11, "driver": "Fernando Alonso", "team": "Aston Martin", "q1": "1:32.121", "q2": "1:31.815", "q3": null},
        {"pos": 12, "driver": "Oliver Bearman", "team": "Haas", "q1": "1:32.269", "q2": "1:31.978", "q3": null},
        {"pos": 13, "driver": "Carlos Sainz", "team": "Williams", "q1": "1:32.457", "q2": "1:32.325", "q3": null},
        {"pos": 14, "driver": "Gabriel Bortoleto", "team": "Kick Sauber", "q1": "1:32.539", "q2": "1:32.564", "q3": null},
        {"pos": 15, "driver": "Isack Hadjar", "team": "Racing Bulls", "q1": "1:32.171", "q2": "DNF", "q3": null},
        {"pos": 16, "driver": "Jack Doohan", "team": "Alpine", "q1": "1:32.575", "q2": null, "q3": null},
        {"pos": 17, "driver": "Pierre Gasly", "team": "Alpine", "q1": "1:32.640", "q2": null, "q3": null},
        {"pos": 18, "driver": "Esteban Ocon", "team": "Haas", "q1": "1:32.651", "q2": null, "q3": null},
        {"pos": 19, "driver": "Nico Hulkenberg", "team": "Kick Sauber", "q1": "1:32.675", "q2": null, "q3": null},
        {"pos": 20, "driver": "Liam Lawson", "team": "Red Bull", "q1": "1:32.729", "q2": null, "q3": null}
      ],
      "sprint": [
        {"pos": 1, "driver": "Lewis Hamilton", "team": "Ferrari", "start": 1, "laps": 19, "points": 8},
        {"pos": 2, "driver": "Oscar Piastri", "team": "McLaren", "start": 3, "laps": 19, "points": 7},
        {"pos": 3, "driver": "Max Verstappen", "team": "Red Bull", "start": 2, "laps": 19, "points": 6},
        {"pos": 4, "driver": "George Russell", "team": "Mercedes", "start": 5, "laps": 19, "points": 5},
        {"pos": 5, "driver": "Charles Leclerc", "team": "Ferrari", "start": 4, "laps": 19, "points": 4},
        {"pos": 6, "driver": "Yuki Tsunoda", "team": "Racing Bulls", "start": 8, "laps": 19, "points": 3},
        {"pos": 7, "driver": "Kimi Antonelli", "team": "Mercedes", "start": 7, "laps": 19, "points": 2},
        {"pos": 8, "driver": "Lando Norris", "team": "McLaren", "start": 6, "laps": 19, "points": 1},
        {"pos": 9, "driver": "Lance Stroll", "team": "Aston Martin", "start": 10, "laps": 19, "points": 0},
        {"pos": 10, "driver": "Fernando Alonso", "team": "Aston Martin", "start": 11, "laps": 19, "points": 0},
        {"pos": 11, "driver": "Alexander Albon", "team": "Williams", "start": 9, "laps": 19, "points": 0},
        {"pos": 12, "driver": "Pierre Gasly", "team": "Alpine", "start": 17, "laps": 19, "points": 0},
        {"pos": 13, "driver": "Isack Hadjar", "team": "Racing Bulls", "start": 15, "laps": 19, "points": 0},
        {"pos": 14, "driver": "Liam Lawson", "team": "Red Bull", "start": 20, "laps": 19, "points": 0},
        {"pos": 15, "driver": "Oliver Bearman", "team": "Haas", "start": 12, "laps": 19, "points": 0},
        {"pos": 16, "driver": "Esteban Ocon", "team": "Haas", "start": 18, "laps": 19, "points": 0},
        {"pos": 17, "driver": "Carlos Sainz", "team": "Williams", "start": 13, "laps": 19, "points": 0},
        {"pos": 18, "driver": "Gabriel Bortoleto", "team": "Kick Sauber", "start": 14, "laps": 19, "points": 0},
        {"pos": 19, "driver": "Nico Hulkenberg", "team": "Kick Sauber", "start": 19, "laps": 19, "points": 0},
        {"pos": 20, "driver": "Jack Doohan", "team": "Alpine", "start": 16, "laps": 19, "points": 0}
      ],
      "qualifying": [
        {"pos": 1, "driver": "Oscar Piastri", "team": "McLaren", "q1": "1:31.591", "q2": "1:31.200", "q3": "1:30.641"},
        {"pos": 2, "driver": "George Russell", "team": "Mercedes", "q1": "1:31.295", "q2": "1:31.307", "q3": "1:30.723"},
        {"pos": 3, "driver": "Lando Norris", "team": "McLaren", "q1": "1:30.983", "q2": "1:30.787", "q3": "1:30.793"},
        {"pos": 4, "driver": "Max Verstappen", "team": "Red Bull", "q1": "1:31.424", "q2": "1:31.142", "q3": "1:30.817"},
        {"pos": 5, "driver": "Lewis Hamilton", "team": "Ferrari", "q1": "1:31.690", "q2": "1:31.501", "q3": "1:30.927"},
        {"pos": 6, "driver": "Charles Leclerc", "team": "Ferrari", "q1": "1:31.579", "q2": "1:31.450", "q3": "1:31.021"},
        {"pos": 7, "driver": "Isack Hadjar", "team": "Racing Bulls", "q1": "1:31.162", "q2": "1:31.253", "q3": "1:31.079"},
        {"pos": 8, "driver": "Kimi Antonelli", "team": "Mercedes", "q1": "1:31.676", "q2": "1:31.590", "q3": "1:31.103"},
        {"pos": 9, "driver": "Yuki Tsunoda", "team": "Racing Bulls", "q1": "1:31.238", "q2": "1:31.260", "q3": "1:31.638"},
        {"pos": 10, "driver": "Alexander Albon", "team": "Williams", "q1": "1:31.503", "q2": "1:31.595", "q3": "1:31.706"},
        {"pos": 11, "driver": "Esteban Ocon", "team": "Haas", "q1": "1:31.876", "q2": "1:31.625", "q3": null},
        {"pos": 12, "driver": "Nico Hulkenberg", "team": "Kick Sauber", "q1": "1:31.921", "q2": "1:31.632", "q3": null},
        {"pos": 13, "driver": "Fernando Alonso", "team": "Aston Martin", "q1": "1:31.719", "q2": "1:31.688", "q3": null},
        {"pos": 14, "driver": "Lance Stroll", "team": "Aston Martin", "q1": "1:31.923", "q2": "1:31.773", "q3": null},
        {"pos": 15, "driver": "Carlos Sainz", "team": "Williams", "q1": "1:31.628", "q2": "1:31.840", "q3": null},
        {"pos": 16, "driver": "Pierre Gasly", "team": "Alpine", "q1": "1:31.992", "q2": null, "q3": null},
        {"pos": 17, "driver": "Oliver Bearman", "team": "Haas", "q1": "1:32.018", "q2": null, "q3": null},
        {"pos": 18, "driver": "Jack Doohan", "team": "Alpine", "q1": "1:32.092", "q2": null, "q3": null},
        {"pos": 19, "driver": "Gabriel Bortoleto", "team": "Kick Sauber", "q1": "1:32.141", "q2": null, "q3": null},
        {"pos": 20, "driver": "Liam Lawson", "team": "Red Bull", "q1": "1:32.174", "q2": null, "q3": null}
      ],
      "race": [
        {"pos": 1, "driver": "Oscar Piastri", "team": "McLaren", "start": 1, "laps": 56, "points": 25},
        {"pos": 2, "driver": "Lando Norris", "team": "McLaren", "start": 3, "laps": 56, "points": 18},
        {"pos": 3, "driver": "George Russell", "team": "Mercedes", "start": 2, "laps": 56, "points": 15},
        {"pos": 4, "driver": "Max Verstappen", "team": "Red Bull", "start": 4, "laps": 56, "points": 12},
        {"pos": 5, "driver": "Esteban Ocon", "team": "Haas", "start": 11, "laps": 56, "points": 10},
        {"pos": 6, "driver": "Kimi Antonelli", "team": "Mercedes", "start": 8, "laps": 56, "points": 8},
        {"pos": 7, "driver": "Alexander Albon", "team": "Williams", "start": 10, "laps": 56, "points": 6},
        {"pos": 8, "driver": "Oliver Bearman", "team": "Haas", "start": 17, "laps": 56, "points": 4},
        {"pos": 9, "driver": "Lance Stroll", "team": "Aston Martin", "start": 14, "laps": 56, "points": 2},
        {"pos": 10, "driver": "Carlos Sainz", "team": "Williams", "start": 15, "laps": 56, "points": 1},
        {"pos": 11, "driver": "Isack Hadjar", "team": "Racing Bulls", "start": 7, "laps": 56, "points": 0},
        {"pos": 12, "driver": "Liam Lawson", "team": "Red Bull", "start": 20, "laps": 56, "points": 0},
        {"pos": 13, "driver": "Jack Doohan", "team": "Alpine", "start": 18, "laps": 56, "points": 0},
        {"pos": 14, "driver": "Gabriel Bortoleto", "team": "Kick Sauber", "start": 19, "laps": 55, "points": 0},
        {"pos": 15, "driver": "Nico Hulkenberg", "team": "Kick Sauber", "start": 12, "laps": 55, "points": 0},
        {"pos": 16, "driver": "Yuki Tsunoda", "team": "Racing Bulls", "start": 9, "laps": 55, "points": 0},
        {"pos": 17, "driver": "Fernando Alonso", "team": "Aston Martin", "start": 13, "laps": 4, "points": 0},
        {"pos": 18, "driver": "Charles Leclerc", "team": "Ferrari", "start": 6, "laps": 0, "points": 0},
        {"pos": 19, "driver": "Lewis Hamilton", "team": "Ferrari", "start": 5, "laps": 0, "points": 0},
        {"pos": 20, "driver": "Pierre Gasly", "team": "Alpine", "start": 16, "laps": 0, "points": 0}
      ]
    },
    "highlights": [
      "McLaren 1-2 with Piastri taking victory",
      "Norris secures double podium for McLaren",
      "Russell completes podium for Mercedes",
      "Ocon scores shock P5 for Haas",
      "Three drivers disqualified (Leclerc, Hamilton, Gasly)"
    ],
    "pitStops": [
      {
        "driver": "Gabriel Bortoleto",
        "stops": 2,
        "totalTime": 46.175,
        "stopsDetail": [{"lap": 1, "time": 23.385}, {"lap": 26, "time": 22.790}]
      },
      {
        "driver": "Pierre Gasly",
        "stops": 1,
        "totalTime": 23.138,
        "stopsDetail": [{"lap": 10, "time": 23.138}]
      },
      {
        "driver": "Yuki Tsunoda",
        "stops": 3,
        "totalTime": 83.204,
        "stopsDetail": [{"lap": 11, "time": 22.519}, {"lap": 35, "time": 22.583}, {"lap": 46, "time": 38.102}]
      },
      {
        "driver": "Esteban Ocon",
        "stops": 1,
        "totalTime": 23.114,
        "stopsDetail": [{"lap": 11, "time": 23.114}]
      },
      {
        "driver": "Jack Doohan",
        "stops": 1,
        "totalTime": 22.793,
        "stopsDetail": [{"lap": 11, "time": 22.793}]
      },
      {
        "driver": "Kimi Antonelli",
        "stops": 1,
        "totalTime": 22.675,
        "stopsDetail": [{"lap": 12, "time": 22.675}]
      },
      {
        "driver": "Isack Hadjar",
        "stops": 2,
        "totalTime": 45.130,
        "stopsDetail": [{"lap": 12, "time": 22.896}, {"lap": 33, "time": 22.234}]
      },
      {
        "driver": "Lewis Hamilton",
        "stops": 2,
        "totalTime": 44.752,
        "stopsDetail": [{"lap": 13, "time": 22.388}, {"lap": 37, "time": 22.364}]
      },
      {
        "driver": "Max Verstappen",
        "stops": 1,
        "totalTime": 22.454,
        "stopsDetail": [{"lap": 13, "time": 22.454}]
      },
      {
        "driver": "Oscar Piastri",
        "stops": 1,
        "totalTime": 23.938,
        "stopsDetail": [{"lap": 14, "time": 23.938}]
      },
      {
        "driver": "George Russell",
        "stops": 1,
        "totalTime": 22.324,
        "stopsDetail": [{"lap": 14, "time": 22.324}]
      },
      {
        "driver": "Lando Norris",
        "stops": 1,
        "totalTime": 22.213,
        "stopsDetail": [{"lap": 15, "time": 22.213}]
      },
      {
        "driver": "Charles Leclerc",
        "stops": 1,
        "totalTime": 22.151,
        "stopsDetail": [{"lap": 15, "time": 22.151}]
      },
      {
        "driver": "Carlos Sainz",
        "stops": 1,
        "totalTime": 22.721,
        "stopsDetail": [{"lap": 17, "time": 22.721}]
      },
      {
        "driver": "Liam Lawson",
        "stops": 2,
        "totalTime": 44.891,
        "stopsDetail": [{"lap": 18, "time": 22.536}, {"lap": 30, "time": 22.355}]
      },
      {
        "driver": "Alexander Albon",
        "stops": 1,
        "totalTime": 23.362,
        "stopsDetail": [{"lap": 20, "time": 23.362}]
      },
      {
        "driver": "Nico Hulkenberg",
        "stops": 1,
        "totalTime": 22.959,
        "stopsDetail": [{"lap": 20, "time": 22.959}]
      },
      {
        "driver": "Oliver Bearman",
        "stops": 1,
        "totalTime": 22.808,
        "stopsDetail": [{"lap": 26, "time": 22.808}]
      },
      {
        "driver": "Lance Stroll",
        "stops": 1,
        "totalTime": 22.995,
        "stopsDetail": [{"lap": 36, "time": 22.995}]
      }
    ],
    "fastestLaps": [
      {"driver": "Lando Norris", "lapTime": "1:35.454", "lap": 53, "avgSpeed": "205.581 km/h"},
      {"driver": "Max Verstappen", "lapTime": "1:35.488", "lap": 56, "avgSpeed": "205.508 km/h"},
      {"driver": "Oscar Piastri", "lapTime": "1:35.520", "lap": 53, "avgSpeed": "205.439 km/h"},
      {"driver": "Esteban Ocon", "lapTime": "1:35.740", "lap": 56, "avgSpeed": "204.967 km/h"},
      {"driver": "George Russell", "lapTime": "1:35.816", "lap": 55, "avgSpeed": "204.805 km/h"},
      {"driver": "Isack Hadjar", "lapTime": "1:35.868", "lap": 35, "avgSpeed": "204.693 km/h"},
      {"driver": "Yuki Tsunoda", "lapTime": "1:35.871", "lap": 49, "avgSpeed": "204.687 km/h"},
      {"driver": "Gabriel Bortoleto", "lapTime": "1:35.874", "lap": 28, "avgSpeed": "204.681 km/h"},
      {"driver": "Liam Lawson", "lapTime": "1:35.985", "lap": 32, "avgSpeed": "204.444 km/h"},
      {"driver": "Lance Stroll", "lapTime": "1:36.044", "lap": 39, "avgSpeed": "204.318 km/h"},
      {"driver": "Kimi Antonelli", "lapTime": "1:36.046", "lap": 56, "avgSpeed": "204.314 km/h"},
      {"driver": "Alexander Albon", "lapTime": "1:36.254", "lap": 52, "avgSpeed": "203.873 km/h"},
      {"driver": "Oliver Bearman", "lapTime": "1:36.363", "lap": 52, "avgSpeed": "203.642 km/h"},
      {"driver": "Jack Doohan", "lapTime": "1:36.424", "lap": 52, "avgSpeed": "203.513 km/h"},
      {"driver": "Carlos Sainz", "lapTime": "1:36.779", "lap": 50, "avgSpeed": "202.767 km/h"},
      {"driver": "Nico Hulkenberg", "lapTime": "1:37.275", "lap": 35, "avgSpeed": "201.733 km/h"},
      {"driver": "Fernando Alonso", "lapTime": "1:39.256", "lap": 3, "avgSpeed": "197.706 km/h"},
      {"driver": "Lewis Hamilton", "lapTime": "1:35.069", "lap": 41, "avgSpeed": "206.414 km/h"},
      {"driver": "Charles Leclerc", "lapTime": "1:36.157", "lap": 49, "avgSpeed": "204.078 km/h"},
      {"driver": "Pierre Gasly", "lapTime": "1:36.425", "lap": 49, "avgSpeed": "203.511 km/h"}
    ]
},
  "australia": {
    "round": 1,
    "name": "Australian Grand Prix",
    "date": "Mar 14-16, 2025",
    "track": "Albert Park Grand Prix Circuit",
    "trackLength": 5.278,
    "weather": "Dry",
    "safetyCars": 0,
    "winner": {
      "driver": "Lando Norris",
      "team": "McLaren"
    },
    "fastestLap": {
      "driver": "Lando Norris",
      "time": "1:22.167",
      "lap": 43
    },
    "results": {
      "qualifying": [
        {"pos": 1, "driver": "Lando Norris", "team": "McLaren", "q1": "1:15.912", "q2": "1:15.415", "q3": "1:15.096"},
        {"pos": 2, "driver": "Oscar Piastri", "team": "McLaren", "q1": "1:16.062", "q2": "1:15.468", "q3": "1:15.180"},
        {"pos": 3, "driver": "Max Verstappen", "team": "Red Bull", "q1": "1:16.018", "q2": "1:15.565", "q3": "1:15.481"},
        {"pos": 4, "driver": "George Russell", "team": "Mercedes", "q1": "1:15.971", "q2": "1:15.798", "q3": "1:15.546"},
        {"pos": 5, "driver": "Yuki Tsunoda", "team": "RB", "q1": "1:16.225", "q2": "1:16.009", "q3": "1:15.670"},
        {"pos": 6, "driver": "Alexander Albon", "team": "Williams", "q1": "1:16.245", "q2": "1:16.017", "q3": "1:15.737"},
        {"pos": 7, "driver": "Charles Leclerc", "team": "Ferrari", "q1": "1:16.029", "q2": "1:15.827", "q3": "1:15.755"},
        {"pos": 8, "driver": "Lewis Hamilton", "team": "Ferrari", "q1": "1:16.213", "q2": "1:15.919", "q3": "1:15.973"},
        {"pos": 9, "driver": "Pierre Gasly", "team": "Alpine", "q1": "1:16.328", "q2": "1:16.112", "q3": "1:15.980"},
        {"pos": 10, "driver": "Carlos Sainz", "team": "Williams", "q1": "1:16.360", "q2": "1:15.931", "q3": "1:16.062"},
        {"pos": 11, "driver": "Isack Hadjar", "team": "RB", "q1": "1:16.354", "q2": "1:16.175", "q3": null},
        {"pos": 12, "driver": "Fernando Alonso", "team": "Aston Martin", "q1": "1:16.288", "q2": "1:16.453", "q3": null},
        {"pos": 13, "driver": "Lance Stroll", "team": "Aston Martin", "q1": "1:16.369", "q2": "1:16.483", "q3": null},
        {"pos": 14, "driver": "Jack Doohan", "team": "Alpine", "q1": "1:16.315", "q2": "1:16.863", "q3": null},
        {"pos": 15, "driver": "Gabriel Bortoleto", "team": "Sauber", "q1": "1:16.516", "q2": "1:17.520", "q3": null},
        {"pos": 16, "driver": "Kimi Antonelli", "team": "Mercedes", "q1": "1:16.525", "q2": null, "q3": null},
        {"pos": 17, "driver": "Nico Hulkenberg", "team": "Sauber", "q1": "1:16.579", "q2": null, "q3": null},
        {"pos": 18, "driver": "Liam Lawson", "team": "Red Bull", "q1": "1:17.094", "q2": null, "q3": null},
        {"pos": 19, "driver": "Esteban Ocon", "team": "Haas", "q1": "1:17.147", "q2": null, "q3": null},
        {"pos": 20, "driver": "Oliver Bearman", "team": "Haas", "q1": "DNS", "q2": null, "q3": null}
      ],
      "race": [
        {"pos": 1, "driver": "Lando Norris", "team": "McLaren", "start": 1, "laps": 57, "points": 25},
        {"pos": 2, "driver": "Max Verstappen", "team": "Red Bull", "start": 3, "laps": 57, "points": 18},
        {"pos": 3, "driver": "George Russell", "team": "Mercedes", "start": 4, "laps": 57, "points": 15},
        {"pos": 4, "driver": "Kimi Antonelli", "team": "Mercedes", "start": 16, "laps": 57, "points": 12},
        {"pos": 5, "driver": "Alexander Albon", "team": "Williams", "start": 6, "laps": 57, "points": 10},
        {"pos": 6, "driver": "Lance Stroll", "team": "Aston Martin", "start": 13, "laps": 57, "points": 8},
        {"pos": 7, "driver": "Nico Hulkenberg", "team": "Sauber", "start": 17, "laps": 57, "points": 6},
        {"pos": 8, "driver": "Charles Leclerc", "team": "Ferrari", "start": 7, "laps": 57, "points": 4},
        {"pos": 9, "driver": "Oscar Piastri", "team": "McLaren", "start": 2, "laps": 57, "points": 2},
        {"pos": 10, "driver": "Lewis Hamilton", "team": "Ferrari", "start": 8, "laps": 57, "points": 1},
        {"pos": 11, "driver": "Pierre Gasly", "team": "Alpine", "start": 9, "laps": 57, "points": 0},
        {"pos": 12, "driver": "Yuki Tsunoda", "team": "RB", "start": 5, "laps": 57, "points": 0},
        {"pos": 13, "driver": "Esteban Ocon", "team": "Haas", "start": 19, "laps": 57, "points": 0},
        {"pos": 14, "driver": "Oliver Bearman", "team": "Haas", "start": 20, "laps": 57, "points": 0},
        {"pos": 15, "driver": "Liam Lawson", "team": "Red Bull", "start": 18, "laps": 46, "points": 0, "status": "DNF"},
        {"pos": 16, "driver": "Gabriel Bortoleto", "team": "Sauber", "start": 15, "laps": 45, "points": 0, "status": "DNF"},
        {"pos": 17, "driver": "Fernando Alonso", "team": "Aston Martin", "start": 12, "laps": 32, "points": 0, "status": "DNF"},
        {"pos": 18, "driver": "Carlos Sainz", "team": "Williams", "start": 10, "laps": 0, "points": 0, "status": "DNF"},
        {"pos": 19, "driver": "Jack Doohan", "team": "Alpine", "start": 14, "laps": 0, "points": 0, "status": "DNF"},
        {"pos": 20, "driver": "Isack Hadjar", "team": "RB", "start": 11, "laps": 0, "points": 0, "status": "DNS"}
      ]
    },
    "highlights": [
      "Norris wins thrilling battle with Verstappen",
      "McLaren secures 1-2 in qualifying",
      "Russell and Antonelli complete strong Mercedes double-points finish",
      "Albon scores points for Williams after impressive drive",
      "Multiple DNFs shake up midfield"
    ],
    "practice": {
      "p1": [
        {"driver": "Lando Norris", "team": "McLaren", "time": "1:17.252", "laps": 21},
        {"driver": "Carlos Sainz", "team": "Williams", "time": "1:17.401", "laps": 25},
        {"driver": "Charles Leclerc", "team": "Ferrari", "time": "1:17.461", "laps": 21},
        {"driver": "Oscar Piastri", "team": "McLaren", "time": "1:17.670", "laps": 20},
        {"driver": "Max Verstappen", "team": "Red Bull", "time": "1:17.696", "laps": 21},
        {"driver": "Alexander Albon", "team": "Williams", "time": "1:17.713", "laps": 18},
        {"driver": "George Russell", "team": "Mercedes", "time": "1:17.716", "laps": 26},
        {"driver": "Fernando Alonso", "team": "Aston Martin", "time": "1:17.736", "laps": 23},
        {"driver": "Isack Hadjar", "team": "RB", "time": "1:17.847", "laps": 25},
        {"driver": "Lance Stroll", "team": "Aston Martin", "time": "1:18.057", "laps": 20},
        {"driver": "Yuki Tsunoda", "team": "RB", "time": "1:18.061", "laps": 23},
        {"driver": "Lewis Hamilton", "team": "Ferrari", "time": "1:18.071", "laps": 20},
        {"driver": "Jack Doohan", "team": "Alpine", "time": "1:18.232", "laps": 20},
        {"driver": "Kimi Antonelli", "team": "Mercedes", "time": "1:18.390", "laps": 25},
        {"driver": "Gabriel Bortoleto", "team": "Sauber", "time": "1:18.438", "laps": 22},
        {"driver": "Liam Lawson", "team": "Red Bull", "time": "1:18.455", "laps": 22},
        {"driver": "Pierre Gasly", "team": "Alpine", "time": "1:18.505", "laps": 23},
        {"driver": "Nico Hulkenberg", "team": "Sauber", "time": "1:18.586", "laps": 18},
        {"driver": "Esteban Ocon", "team": "Haas", "time": "1:19.139", "laps": 16},
        {"driver": "Oliver Bearman", "team": "Haas", "time": "1:19.312", "laps": 12}
      ],
      "p2": [
        {"driver": "Charles Leclerc", "team": "Ferrari", "time": "1:16.439", "laps": 32},
        {"driver": "Oscar Piastri", "team": "McLaren", "time": "1:16.563", "laps": 30},
        {"driver": "Lando Norris", "team": "McLaren", "time": "1:16.580", "laps": 30},
        {"driver": "Yuki Tsunoda", "team": "RB", "time": "1:16.784", "laps": 29},
        {"driver": "Lewis Hamilton", "team": "Ferrari", "time": "1:16.859", "laps": 31},
        {"driver": "Isack Hadjar", "team": "RB", "time": "1:17.019", "laps": 30},
        {"driver": "Max Verstappen", "team": "Red Bull", "time": "1:17.063", "laps": 22},
        {"driver": "Nico Hulkenberg", "team": "Sauber", "time": "1:17.161", "laps": 24},
        {"driver": "Lance Stroll", "team": "Aston Martin", "time": "1:17.279", "laps": 28},
        {"driver": "George Russell", "team": "Mercedes", "time": "1:17.282", "laps": 30},
        {"driver": "Carlos Sainz", "team": "Williams", "time": "1:17.302", "laps": 30},
        {"driver": "Alexander Albon", "team": "Williams", "time": "1:17.302", "laps": 28},
        {"driver": "Fernando Alonso", "team": "Aston Martin", "time": "1:17.330", "laps": 27},
        {"driver": "Jack Doohan", "team": "Alpine", "time": "1:17.394", "laps": 30},
        {"driver": "Pierre Gasly", "team": "Alpine", "time": "1:17.493", "laps": 30},
        {"driver": "Kimi Antonelli", "team": "Mercedes", "time": "1:17.634", "laps": 31},
        {"driver": "Liam Lawson", "team": "Red Bull", "time": "1:17.640", "laps": 30},
        {"driver": "Gabriel Bortoleto", "team": "Sauber", "time": "1:17.847", "laps": 29},
        {"driver": "Esteban Ocon", "team": "Haas", "time": "1:18.034", "laps": 31}
      ],
      "p3": [
        {"driver": "Oscar Piastri", "team": "McLaren", "time": "1:15.921", "laps": 16},
        {"driver": "George Russell", "team": "Mercedes", "time": "1:15.960", "laps": 17},
        {"driver": "Max Verstappen", "team": "Red Bull", "time": "1:16.002", "laps": 18},
        {"driver": "Charles Leclerc", "team": "Ferrari", "time": "1:16.188", "laps": 22},
        {"driver": "Kimi Antonelli", "team": "Mercedes", "time": "1:16.206", "laps": 20},
        {"driver": "Carlos Sainz", "team": "Williams", "time": "1:16.252", "laps": 23},
        {"driver": "Alexander Albon", "team": "Williams", "time": "1:16.258", "laps": 21},
        {"driver": "Lewis Hamilton", "team": "Ferrari", "time": "1:16.378", "laps": 21},
        {"driver": "Yuki Tsunoda", "team": "RB", "time": "1:16.455", "laps": 17},
        {"driver": "Lando Norris", "team": "McLaren", "time": "1:16.597", "laps": 20},
        {"driver": "Gabriel Bortoleto", "team": "Sauber", "time": "1:16.707", "laps": 18},
        {"driver": "Pierre Gasly", "team": "Alpine", "time": "1:16.719", "laps": 22},
        {"driver": "Isack Hadjar", "team": "RB", "time": "1:16.732", "laps": 18},
        {"driver": "Lance Stroll", "team": "Aston Martin", "time": "1:16.948", "laps": 22},
        {"driver": "Jack Doohan", "team": "Alpine", "time": "1:16.993", "laps": 18},
        {"driver": "Nico Hulkenberg", "team": "Sauber", "time": "1:17.146", "laps": 15},
        {"driver": "Fernando Alonso", "team": "Aston Martin", "time": "1:17.270", "laps": 22},
        {"driver": "Esteban Ocon", "team": "Haas", "time": "1:17.373", "laps": 18},
        {"driver": "Oliver Bearman", "team": "Haas", "time": "No time", "laps": 2},
        {"driver": "Liam Lawson", "team": "Red Bull", "time": "No time", "laps": 2}
      ]
    },
    "pitStops": [
      {
        "driver": "Lando Norris",
        "stops": 3,
        "totalTime": 39.055,
        "stopsDetail": [
          {"lap": 2, "time": "13.341"},
          {"lap": 3, "time": "12.910"},
          {"lap": 4, "time": "12.804"}
        ]
      },
      {
        "driver": "Max Verstappen",
        "stops": 3,
        "totalTime": 40.094,
        "stopsDetail": [
          {"lap": 2, "time": "13.416"},
          {"lap": 3, "time": "13.740"},
          {"lap": 4, "time": "12.938"}
        ]
      },
      {
        "driver": "George Russell",
        "stops": 3,
        "totalTime": 41.472,
        "stopsDetail": [
          {"lap": 2, "time": "13.672"},
          {"lap": 3, "time": "14.104"},
          {"lap": 4, "time": "13.696"}
        ]
      },
      {
        "driver": "Charles Leclerc",
        "stops": 3,
        "totalTime": 42.932,
        "stopsDetail": [
          {"lap": 2, "time": "14.182"},
          {"lap": 3, "time": "14.621"},
          {"lap": 4, "time": "14.129"}
        ]
      },
      {
        "driver": "Yuki Tsunoda",
        "stops": 4,
        "totalTime": 62.782,
        "stopsDetail": [
          {"lap": 2, "time": "14.957"},
          {"lap": 3, "time": "15.223"},
          {"lap": 4, "time": "13.605"},
          {"lap": 33, "time": "18.997"}
        ]
      }
    ],
    "fastestLaps": [
      {"driver": "Lando Norris", "lapTime": "1:22.167", "lap": 43, "avgSpeed": "231.246 km/h"},
      {"driver": "Liam Lawson", "lapTime": "1:22.970", "lap": 43, "avgSpeed": "229.008 km/h"},
      {"driver": "Max Verstappen", "lapTime": "1:23.081", "lap": 43, "avgSpeed": "228.702 km/h"},
      {"driver": "Oscar Piastri", "lapTime": "1:23.242", "lap": 43, "avgSpeed": "228.259 km/h"},
      {"driver": "Gabriel Bortoleto", "lapTime": "1:24.192", "lap": 43, "avgSpeed": "225.684 km/h"},
      {"driver": "Yuki Tsunoda", "lapTime": "1:24.194", "lap": 43, "avgSpeed": "225.678 km/h"},
      {"driver": "Lewis Hamilton", "lapTime": "1:24.218", "lap": 43, "avgSpeed": "225.614 km/h"},
      {"driver": "Alexander Albon", "lapTime": "1:24.597", "lap": 43, "avgSpeed": "224.603 km/h"},
      {"driver": "Kimi Antonelli", "lapTime": "1:24.901", "lap": 43, "avgSpeed": "223.799 km/h"},
      {"driver": "Pierre Gasly", "lapTime": "1:25.020", "lap": 43, "avgSpeed": "223.486 km/h"},
      {"driver": "George Russell", "lapTime": "1:25.065", "lap": 43, "avgSpeed": "223.368 km/h"},
      {"driver": "Nico Hulkenberg", "lapTime": "1:25.243", "lap": 43, "avgSpeed": "222.901 km/h"},
      {"driver": "Charles Leclerc", "lapTime": "1:25.271", "lap": 43, "avgSpeed": "222.828 km/h"},
      {"driver": "Lance Stroll", "lapTime": "1:25.538", "lap": 43, "avgSpeed": "222.132 km/h"},
      {"driver": "Esteban Ocon", "lapTime": "1:26.764", "lap": 42, "avgSpeed": "218.994 km/h"},
      {"driver": "Oliver Bearman", "lapTime": "1:27.603", "lap": 42, "avgSpeed": "216.896 km/h"},
      {"driver": "Fernando Alonso", "lapTime": "1:28.819", "lap": 32, "avgSpeed": "213.927 km/h"}
    ]
  }
};