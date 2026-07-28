const { execSync } = require("child_process");
const { version } = require("process");

try {
  const pyVersion = execSync("python --version", { encoding: "utf8" }).trim();
  const match = pyVersion.match(/Python (\d+)\.(\d+)/);
  if (!match || parseInt(match[1]) < 3 || (parseInt(match[1]) === 3 && parseInt(match[2]) < 10)) {
    console.error("Error: Python >= 3.10 is required. Found: " + pyVersion);
    process.exit(1);
  }
  console.log("✓ " + pyVersion + " detected");
} catch {
  console.error("Error: Python not found. Install Python >= 3.10 first.");
  process.exit(1);
}

try {
  execSync("python -c \"import src.inference.cli\" 2>nul || pip install -e " + __dirname + "/../..", {
    stdio: "inherit",
    cwd: __dirname + "/../..",
  });
  console.log("✓ Nexus package installed");
} catch {
  console.log("ℹ Run 'pip install -e .' in the nexus repo to install the Python package");
}