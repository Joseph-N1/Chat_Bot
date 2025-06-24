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
}
};