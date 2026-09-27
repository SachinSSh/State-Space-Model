const fs = require('fs');
const path = require('path');

const appFile = path.join(__dirname, 'src', 'App.tsx.backup');
const content = fs.readFileSync(appFile, 'utf8');

// The file has imports at the top, then `function App() { return ( <div ...> ... </div> ); } export default App;`

function extractBetween(str, startToken, endToken) {
    const startIndex = str.indexOf(startToken);
    if (startIndex === -1) return null;
    const endIndex = endToken ? str.indexOf(endToken, startIndex) : str.length;
    if (endIndex === -1) return null;
    return str.substring(startIndex, endIndex);
}

const importsBlock = content.substring(0, content.indexOf('function App()'));

const navbar = extractBetween(content, '{/* Navbar */}', '{/* Hero Section */}');
const hero = extractBetween(content, '{/* Hero Section */}', '{/* Core Pillars */}');
const pillars = extractBetween(content, '{/* Core Pillars */}', '{/* Architecture & Lineage */}');
const architecture = extractBetween(content, '{/* Architecture & Lineage */}', '{/* Model Deep Dives */}');
const modelDeepDives = extractBetween(content, '{/* Model Deep Dives */}', '{/* Inference & Handoff */}');
const inference = extractBetween(content, '{/* Inference & Handoff */}', '{/* Scaling & Real Text Training */}');
const scaling = extractBetween(content, '{/* Scaling & Real Text Training */}', '{/* Data Pipeline */}');
const dataPipeline = extractBetween(content, '{/* Data Pipeline */}', '{/* The 14-Model Roadmap */}');
const roadmap = extractBetween(content, '{/* The 14-Model Roadmap */}', '{/* Codebase Architecture */}');
const codebase = extractBetween(content, '{/* Codebase Architecture */}', '{/* Benchmarks */}');
const benchmarks = extractBetween(content, '{/* Benchmarks */}', '{/* Test Suite */}');
const testSuite = extractBetween(content, '{/* Test Suite */}', '{/* Notebooks */}');
const notebooks = extractBetween(content, '{/* Notebooks */}', '{/* References / Papers */}');
const references = extractBetween(content, '{/* References / Papers */}', '{/* Quickstart */}');
const quickstart = extractBetween(content, '{/* Quickstart */}', '{/* Footer */}');
const footer = extractBetween(content, '{/* Footer */}', '    </div>\n  );\n}');

const navLinks = `
            <Link to="/" className="text-sm font-medium hover:text-white transition-colors whitespace-nowrap">Home</Link>
            <Link to="/models" className="text-sm font-medium hover:text-white transition-colors whitespace-nowrap">Models</Link>
            <Link to="/benchmarks" className="text-sm font-medium hover:text-white transition-colors whitespace-nowrap">Benchmarks</Link>
            <a href="https://github.com/SachinSSh/State-Space-Model" target="_blank" rel="noopener noreferrer" className="text-sm font-medium hover:text-white transition-colors whitespace-nowrap">GitHub</a>
`;

let newNavbar = navbar.replace(
    /<div className="flex items-center gap-6 overflow-x-auto">[\s\S]*?<\/div>/,
    `<div className="flex items-center gap-6 overflow-x-auto">${navLinks}</div>`
);

const layoutComponent = `
import { Link, Outlet } from 'react-router-dom';
import { Layers } from 'lucide-react';

export default function Layout() {
  return (
    <div className="min-h-screen bg-[#0B0F19] text-slate-300 font-sans selection:bg-blue-500/30">
      ${newNavbar}
      <div className="pt-16">
        <Outlet />
      </div>
      ${footer}
    </div>
  );
}
`;
fs.writeFileSync(path.join(__dirname, 'src', 'components', 'Layout.tsx'), layoutComponent);

const heroFixed = hero.replace(/<a href="#models"/g, '<Link to="/models"').replace(/<\/a>/g, '</Link>').replace(/<a href="#quickstart"/g, '<a href="/#quickstart"');

const homeComponent = `
import { Terminal, ArrowRight, Activity, GitBranch, Cpu, Zap, Database } from 'lucide-react';
import { Link } from 'react-router-dom';
import { InlineMath, BlockMath } from 'react-katex';

export default function HomePage() {
  return (
    <>
      ${heroFixed}
      ${pillars}
      ${architecture}
      ${inference}
      ${scaling}
      ${dataPipeline}
      ${codebase}
      ${quickstart}
    </>
  );
}
`;
fs.writeFileSync(path.join(__dirname, 'src', 'pages', 'HomePage.tsx'), homeComponent);

const modelsComponent = `
import { CheckCircle2, ShieldCheck } from 'lucide-react';
import { InlineMath, BlockMath } from 'react-katex';

export default function ModelsPage() {
  return (
    <>
      ${modelDeepDives}
      ${roadmap}
    </>
  );
}
`;
fs.writeFileSync(path.join(__dirname, 'src', 'pages', 'ModelsPage.tsx'), modelsComponent);

const benchmarksComponent = `
import { InlineMath, BlockMath } from 'react-katex';

export default function BenchmarksPage() {
  return (
    <>
      ${benchmarks}
      ${testSuite}
      ${notebooks}
      ${references}
    </>
  );
}
`;
fs.writeFileSync(path.join(__dirname, 'src', 'pages', 'BenchmarksPage.tsx'), benchmarksComponent);

const newApp = `
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import 'katex/dist/katex.min.css';
import Layout from './components/Layout';
import HomePage from './pages/HomePage';
import ModelsPage from './pages/ModelsPage';
import BenchmarksPage from './pages/BenchmarksPage';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<HomePage />} />
          <Route path="models" element={<ModelsPage />} />
          <Route path="benchmarks" element={<BenchmarksPage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

export default App;
`;
fs.writeFileSync(path.join(__dirname, 'src', 'App.tsx'), newApp);

console.log("Splitting complete!");
