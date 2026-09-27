
import { Link, Outlet } from 'react-router-dom';
import { Layers } from 'lucide-react';

export default function Layout() {
  return (
    <div className="min-h-screen bg-[#0B0F19] text-slate-300 font-sans selection:bg-blue-500/30">
      {/* Navbar */}
      <nav className="fixed top-0 w-full z-50 bg-[#0B0F19]/80 backdrop-blur-xl border-b border-white/10">
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
          <Link to="/" className="flex items-center gap-3 hover:opacity-80 transition-opacity">
            <div className="bg-blue-500 p-1.5 rounded-lg">
              <Layers className="text-white w-5 h-5" />
            </div>
            <span className="font-bold text-lg text-white tracking-tight">SSM Lab</span>
          </Link>
          <div className="flex items-center gap-6 overflow-x-auto">
            <Link to="/" className="text-sm font-medium hover:text-white transition-colors whitespace-nowrap">Home</Link>
            <Link to="/models" className="text-sm font-medium hover:text-white transition-colors whitespace-nowrap">Models</Link>
            <Link to="/benchmarks" className="text-sm font-medium hover:text-white transition-colors whitespace-nowrap">Benchmarks</Link>
            <a href="https://github.com/SachinSSh/State-Space-Model" target="_blank" rel="noopener noreferrer" className="text-sm font-medium hover:text-white transition-colors whitespace-nowrap">GitHub</a>
</div>
        </div>
      </nav>

      
      <div className="pt-16">
        <Outlet />
      </div>
      {/* Footer */}
      <footer className="py-16 px-6 border-t border-white/5">
        <div className="max-w-7xl mx-auto">
          <div className="grid md:grid-cols-4 gap-12 mb-12">
            <div>
              <Link to="/" className="flex items-center gap-2 mb-4 hover:opacity-80 transition-opacity">
                <div className="bg-blue-500 p-1.5 rounded-lg">
                  <Layers className="text-white w-4 h-4" />
                </div>
                <span className="font-bold text-white">SSM Lab</span>
              </Link>
              <p className="text-sm text-slate-500 leading-relaxed">
                A from-scratch comparative study of sequence architectures. Built to be compared, not just demonstrated.
              </p>
            </div>
            <div>
              <h4 className="font-bold text-white text-sm uppercase tracking-wider mb-4">Navigate</h4>
              <ul className="space-y-2 text-sm text-slate-500">
                <li><Link to="/#architecture" className="hover:text-white transition-colors">Architecture</Link></li>
                <li><Link to="/models" className="hover:text-white transition-colors">Model Deep Dives</Link></li>
                <li><Link to="/benchmarks" className="hover:text-white transition-colors">Benchmarks</Link></li>
                <li><Link to="/#scaling" className="hover:text-white transition-colors">Scaling</Link></li>
                <li><Link to="/#codebase" className="hover:text-white transition-colors">Codebase</Link></li>
                <li><Link to="/#quickstart" className="hover:text-white transition-colors">Quickstart</Link></li>
              </ul>
            </div>
            <div>
              <h4 className="font-bold text-white text-sm uppercase tracking-wider mb-4">Project Stats</h4>
              <ul className="space-y-2 text-sm text-slate-500">
                <li>14 models scoped</li>
                <li>2 models shipped</li>
                <li>6 benchmark tasks</li>
                <li>13 test files</li>
                <li>3 interactive notebooks</li>
              </ul>
            </div>
            <div>
              <h4 className="font-bold text-white text-sm uppercase tracking-wider mb-4">Links</h4>
              <ul className="space-y-2 text-sm text-slate-500">
                <li><a href="https://github.com/SachinSSh/State-Space-Model" className="hover:text-white transition-colors">GitHub Repository</a></li>
                <li><a href="https://github.com/SachinSSh/State-Space-Model/blob/main/ROADMAP.md" className="hover:text-white transition-colors">Roadmap</a></li>
                <li><a href="https://github.com/SachinSSh/State-Space-Model/blob/main/docs/00_research_lineage.md" className="hover:text-white transition-colors">Research Lineage</a></li>
                <li><a href="https://github.com/SachinSSh/State-Space-Model/blob/main/docs/how_to_add_a_model.md" className="hover:text-white transition-colors">Contributing Guide</a></li>
              </ul>
            </div>
          </div>
          <div className="border-t border-white/5 pt-8 text-center text-slate-600 text-xs">
            <p>© 2026 State-Space Models Comparative Study. MIT Licensed. Built with PyTorch, React, and genuine rigor.</p>
          </div>
        </div>
      </footer>

    </div>
  );
}
