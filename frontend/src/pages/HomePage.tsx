
import { Terminal, ArrowRight, Activity, GitBranch, Cpu, Zap, Database } from 'lucide-react';
import { Link } from 'react-router-dom';
import { InlineMath } from 'react-katex';

export default function HomePage() {
  return (
    <>
      {/* Hero Section */}
      <section className="pt-32 pb-24 px-6 relative overflow-hidden flex flex-col items-center justify-center min-h-[90vh]">
        {/* Animated Background Gradients & Grid */}
        <div className="absolute top-1/4 left-1/4 -translate-x-1/2 w-full max-w-xl h-[400px] bg-blue-600/30 blur-[120px] rounded-full pointer-events-none animate-float"></div>
        <div className="absolute top-1/3 right-1/4 translate-x-1/4 w-full max-w-lg h-[350px] bg-purple-600/20 blur-[100px] rounded-full pointer-events-none animate-float-delayed"></div>
        <div className="absolute inset-0 bg-[linear-gradient(to_right,#80808012_1px,transparent_1px),linear-gradient(to_bottom,#80808012_1px,transparent_1px)] bg-[size:24px_24px] [mask-image:radial-gradient(ellipse_60%_50%_at_50%_50%,#000_70%,transparent_100%)] pointer-events-none"></div>

        <div className="max-w-5xl mx-auto text-center relative z-10 w-full flex flex-col items-center">
          <div className="animate-fade-up">
            <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-blue-500/10 border border-blue-500/30 text-blue-400 text-sm font-bold tracking-wide mb-10 shadow-[0_0_20px_rgba(59,130,246,0.2)] backdrop-blur-md">
              <Activity className="w-4 h-4 animate-pulse" />
              <span>2 of 14 Models Shipped</span>
            </div>
          </div>
          
          <h1 className="text-6xl md:text-8xl font-extrabold text-white tracking-tight mb-8 leading-[1.1] animate-fade-up delay-100 drop-shadow-2xl">
            Beyond Attention.<br />
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 via-blue-500 to-purple-500 animate-gradient-x inline-block mt-2">
              Built to be Compared.
            </span>
          </h1>
          
          <p className="text-xl md:text-2xl text-slate-400 mb-12 max-w-3xl mx-auto leading-relaxed animate-fade-up delay-200 font-light">
            A from-scratch, independently-verified comparative study of sequence architectures challenging attention. 
            <strong className="text-white font-medium"> Every repo shows a loss curve, but none show the <i className="italic">same</i> curve. We do.</strong>
          </p>
          
          <div className="flex flex-col sm:flex-row items-center justify-center gap-6 animate-fade-up delay-300 w-full sm:w-auto">
            <a href="/#quickstart" className="w-full sm:w-auto flex items-center justify-center gap-3 bg-white text-slate-900 px-8 py-4 rounded-xl font-extrabold hover:bg-slate-100 hover:scale-105 transition-all shadow-[0_0_30px_rgba(255,255,255,0.3)] hover:shadow-[0_0_40px_rgba(255,255,255,0.5)]">
              <Terminal className="w-5 h-5" />
              Quickstart Guide
            </a>
            <Link to="/models" className="w-full sm:w-auto flex items-center justify-center gap-3 bg-[#1A2333]/80 backdrop-blur-md border border-white/10 text-white px-8 py-4 rounded-xl font-bold hover:bg-[#232D40] hover:border-white/20 transition-all group shadow-lg">
              Read the Deep Dives
              <ArrowRight className="w-5 h-5 group-hover:translate-x-1.5 transition-transform" />
            </Link>
          </div>
        </div>
      </section>

      
      {/* Animated Stats Bar */}
      <section className="py-12 px-6 border-y border-white/5 bg-[#0B0F19] relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-r from-blue-500/5 via-transparent to-purple-500/5 pointer-events-none"></div>
        <div className="max-w-7xl mx-auto grid grid-cols-2 md:grid-cols-4 gap-8 text-center relative z-10">
          <div className="animate-fade-up delay-100">
            <div className="text-4xl md:text-5xl font-extrabold text-white mb-1">14</div>
            <div className="text-sm text-slate-500 uppercase tracking-widest font-medium">Models Scoped</div>
          </div>
          <div className="animate-fade-up delay-200">
            <div className="text-4xl md:text-5xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-green-400 to-emerald-400 mb-1">2</div>
            <div className="text-sm text-slate-500 uppercase tracking-widest font-medium">Shipped & Verified</div>
          </div>
          <div className="animate-fade-up delay-300">
            <div className="text-4xl md:text-5xl font-extrabold text-white mb-1">6</div>
            <div className="text-sm text-slate-500 uppercase tracking-widest font-medium">Benchmark Tasks</div>
          </div>
          <div className="animate-fade-up delay-400">
            <div className="text-4xl md:text-5xl font-extrabold text-white mb-1">13</div>
            <div className="text-sm text-slate-500 uppercase tracking-widest font-medium">Test Files</div>
          </div>
        </div>
      </section>

      {/* Core Pillars */}
      <section className="py-24 px-6 bg-[#0D1321]">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16 animate-fade-up">
            <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">Why This Repository Exists</h2>
            <p className="text-lg text-slate-400 max-w-2xl mx-auto">Three pillars that set this project apart from every other SSM implementation.</p>
          </div>
          <div className="grid md:grid-cols-3 gap-6">
            <div className="group bg-[#151C2C] border border-white/5 p-8 rounded-2xl hover:border-blue-500/30 hover:shadow-[0_0_30px_rgba(59,130,246,0.1)] transition-all duration-300 animate-fade-up delay-100">
              <div className="w-12 h-12 bg-blue-500/10 rounded-xl flex items-center justify-center mb-6 border border-blue-500/20 group-hover:bg-blue-500/20 group-hover:scale-110 transition-all duration-300">
                <Cpu className="text-blue-400 w-6 h-6" />
              </div>
              <h3 className="text-xl font-bold text-white mb-3">One Shared Backbone</h3>
              <p className="text-slate-400 text-sm leading-relaxed">
                14 architectures, one shared residual backbone. S4D, Mamba, RWKV-7, and hybrids all use the exact same stack. The only thing that changes is the sequence mixer.
              </p>
            </div>
            <div className="group bg-[#151C2C] border border-white/5 p-8 rounded-2xl hover:border-purple-500/30 hover:shadow-[0_0_30px_rgba(168,85,247,0.1)] transition-all duration-300 animate-fade-up delay-200">
              <div className="w-12 h-12 bg-purple-500/10 rounded-xl flex items-center justify-center mb-6 border border-purple-500/20 group-hover:bg-purple-500/20 group-hover:scale-110 transition-all duration-300">
                <Database className="text-purple-400 w-6 h-6" />
              </div>
              <h3 className="text-xl font-bold text-white mb-3">Six Strict Benchmarks</h3>
              <p className="text-slate-400 text-sm leading-relaxed">
                MQAR, Copying, Induction heads, Parity tracking, LRA, and real LM perplexity. Chosen specifically to expose structural limits, not luck with hyperparameters.
              </p>
            </div>
            <div className="group bg-[#151C2C] border border-white/5 p-8 rounded-2xl hover:border-green-500/30 hover:shadow-[0_0_30px_rgba(34,197,94,0.1)] transition-all duration-300 animate-fade-up delay-300">
              <div className="w-12 h-12 bg-green-500/10 rounded-xl flex items-center justify-center mb-6 border border-green-500/20 group-hover:bg-green-500/20 group-hover:scale-110 transition-all duration-300">
                <Zap className="text-green-400 w-6 h-6" />
              </div>
              <h3 className="text-xl font-bold text-white mb-3">Deployable Inference</h3>
              <p className="text-slate-400 text-sm leading-relaxed">
                Companion project (ssm-infer) ships trained models behind a running inference server, proving a constant O(1) memory footprint during generation.
              </p>
            </div>
          </div>
        </div>
      </section>

      
      {/* Architecture & Lineage */}
      <section id="architecture" className="py-24 px-6 relative">
        <div className="max-w-7xl mx-auto">
          <div className="mb-16 text-center">
            <h2 className="text-3xl md:text-5xl font-bold text-white mb-6">The Universal Scaffolding</h2>
            <p className="text-lg text-slate-400 max-w-2xl mx-auto">
              Before any of the 14 architectures, we answered one question: is there a single shape general enough for all of them? Yes.
            </p>
          </div>

          <div className="grid lg:grid-cols-2 gap-12 items-center mb-20">
            <div className="bg-[#151C2C] p-2 rounded-2xl border border-white/5 shadow-2xl">
              <img src="/assets/diagrams/01-architecture-overview.png" alt="Architecture" className="w-full rounded-xl" />
            </div>
            <div>
              <div className="flex items-center gap-3 mb-4">
                <GitBranch className="text-blue-400" />
                <h3 className="text-2xl font-bold text-white">The Shape Every Model Shares</h3>
              </div>
              <p className="text-slate-400 mb-6 leading-relaxed">
                Every state space model in this repository reduces to an input map, a linear recurrent core (<InlineMath math="\bar{A}, \bar{B}, \bar{C}, \bar{D}" />), and a nonlinear output gate.
              </p>
              <div className="bg-[#0B0F19] border border-white/5 rounded-xl p-6 font-mono text-sm text-slate-300">
                <span className="text-purple-400">embed</span>
                <br/>
                &nbsp;&nbsp;→ [ norm → <span className="text-blue-400">mixer</span> → residual → norm → FFN → residual ] × N
                <br/>
                &nbsp;&nbsp;→ final norm → <span className="text-green-400">head</span>
              </div>
              <p className="text-sm text-slate-500 mt-4">
                The "mixer" is a <code className="text-blue-400 bg-blue-400/10 px-1 py-0.5 rounded">SequenceMixer</code> subclass implementing exactly the recurrent core. Nothing else changes.
              </p>
            </div>
          </div>

          <div className="grid lg:grid-cols-2 gap-12 items-center flex-row-reverse">
            <div className="order-2 lg:order-1">
              <h3 className="text-2xl font-bold text-white mb-4">The Parallel Scan Primitive</h3>
              <p className="text-slate-400 mb-6 leading-relaxed">
                Nearly every model on the roadmap is, underneath its own notation, the same recurrence: <InlineMath math="h_t = a_t \cdot h_{t-1} + b_t" />.
              </p>
              <p className="text-slate-400 mb-6 leading-relaxed">
                One scan primitive, <code className="text-pink-400 bg-pink-400/10 px-1.5 py-0.5 rounded">parallel_scan_log</code>, evaluates all of them in <InlineMath math="O(\log L)" /> sequential rounds instead of <InlineMath math="L" />. It is verified against a plain sequential ground truth to <code className="text-white">1e-4</code> for both real and complex numbers.
              </p>
            </div>
            <div className="order-1 lg:order-2 bg-[#151C2C] p-2 rounded-2xl border border-white/5 shadow-2xl">
              <img src="/assets/diagrams/03-parallel-scan.png" alt="Parallel Scan" className="w-full rounded-xl" />
            </div>
          </div>
        </div>
      </section>

      
      {/* Inference & Handoff */}
      <section className="py-24 px-6 relative">
        <div className="max-w-7xl mx-auto grid lg:grid-cols-2 gap-16 items-center">
          <div>
            <h2 className="text-3xl md:text-5xl font-bold text-white mb-6">Project B: Inference</h2>
            <p className="text-lg text-slate-400 mb-6 leading-relaxed">
              A repo that only produces loss curves hasn't answered "does this work." So we ship a separate, independent project: <strong>ssm-infer</strong>.
            </p>
            <p className="text-slate-400 mb-8 leading-relaxed">
              Weights are stripped via <code className="text-blue-400 bg-blue-400/10 px-1 py-0.5 rounded">export_for_inference.py</code> and loaded independently. Parity tests ensure the inference forward pass exactly matches the training pass.
            </p>
            
            <div className="bg-[#151C2C] rounded-2xl border border-white/5 p-6 shadow-xl">
              <h4 className="text-white font-bold mb-4">O(1) Inference Footprint Proved</h4>
              <p className="text-sm text-slate-400 mb-4">Measured generation on the exported model. Unlike a Transformer KV cache which grows with length, the recurrent state is fixed.</p>
              
              <div className="overflow-hidden rounded-xl border border-white/10">
                <table className="w-full text-left text-sm text-slate-300">
                  <thead className="bg-white/5 border-b border-white/10">
                    <tr>
                      <th className="p-3 font-semibold">Tokens</th>
                      <th className="p-3 font-semibold">State Size</th>
                      <th className="p-3 font-semibold">Time/Token</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-white/5 bg-[#0B0F19]">
                    <tr><td className="p-3">10</td><td className="p-3">18,432 bytes</td><td className="p-3 font-mono text-blue-400">1.09 ms</td></tr>
                    <tr><td className="p-3">1,000</td><td className="p-3">18,432 bytes</td><td className="p-3 font-mono text-blue-400">1.21 ms</td></tr>
                    <tr><td className="p-3">2,000</td><td className="p-3">18,432 bytes</td><td className="p-3 font-mono text-blue-400">1.06 ms</td></tr>
                  </tbody>
                </table>
              </div>
            </div>
          </div>
          
          <div className="bg-[#151C2C] p-2 rounded-3xl border border-white/5 shadow-2xl">
            <img src="/assets/diagrams/06-project-handoff.png" alt="Project Handoff" className="w-full rounded-2xl" />
          </div>
        </div>
      </section>

      
      {/* Scaling & Real Text Training */}
      <section id="scaling" className="py-24 px-6 border-t border-white/5 bg-[#0B0F19]">
        <div className="max-w-7xl mx-auto">
          <div className="mb-16 text-center">
            <h2 className="text-3xl md:text-5xl font-bold text-white mb-6">Scaling & Real Text</h2>
            <p className="text-lg text-slate-400 max-w-2xl mx-auto">
              From synthetic probes to actual multi-session training runs on real text corpora.
            </p>
          </div>

          <div className="grid lg:grid-cols-2 gap-12 items-center mb-16">
            <div>
              <h3 className="text-2xl font-bold text-white mb-4">Gradient Checkpointing</h3>
              <p className="text-slate-400 mb-6 leading-relaxed">
                Recomputes each layer's activations during backward instead of storing them. A plain config (d_model=64, 3 layers) <strong>OOM-killed the process</strong> in our 4GB sandbox. The identical config with <code className="bg-slate-800 text-pink-400 px-1 py-0.5 rounded text-sm">use_gradient_checkpointing=True</code> completed successfully. We measured a ~40% reduction in peak RSS on moderate configs.
              </p>
              <h3 className="text-2xl font-bold text-white mb-4">Gradient Accumulation</h3>
              <p className="text-slate-400 mb-6 leading-relaxed">
                Runs <code className="text-white">k</code> micro-batches through forward/backward before an optimizer step. Verified mathematically equivalent to one <code className="text-white">k</code>-times-larger batch down to <code className="text-blue-300">~1e-8</code> precision.
              </p>
            </div>
            <div className="bg-[#151C2C] p-2 rounded-2xl border border-white/5 shadow-2xl">
              <img src="/assets/diagrams/14-gradient-checkpointing-memory.png" alt="Gradient Checkpointing Memory" className="w-full rounded-xl" />
            </div>
          </div>

          <div className="grid lg:grid-cols-2 gap-12 items-center mb-20 flex-row-reverse">
            <div className="order-2 lg:order-1 bg-[#151C2C] p-2 rounded-2xl border border-white/5 shadow-2xl">
              <img src="/assets/diagrams/15-checkpoint-resume-timeline.png" alt="Checkpoint Resume Timeline" className="w-full rounded-xl" />
            </div>
            <div className="order-1 lg:order-2">
              <h3 className="text-2xl font-bold text-white mb-4">Multi-Session Checkpoint & Resume</h3>
              <p className="text-slate-400 mb-6 leading-relaxed">
                Exact model, optimizer (AdamW moment estimates), and RNG-stream restoration across sessions. Verified by reloading into a differently-initialized model and checking bit-identical eval loss. This is exactly what let the 69K-parameter Shakespeare model get trained in 5 separate sessions.
              </p>
              <div className="bg-[#151C2C] rounded-xl border border-white/5 overflow-hidden">
                <table className="w-full text-left text-sm text-slate-300">
                  <thead className="bg-white/5 border-b border-white/10">
                    <tr><th className="p-3">Preset</th><th className="p-3">Params</th><th className="p-3">Checkpointing</th><th className="p-3">Accumulation</th></tr>
                  </thead>
                  <tbody className="divide-y divide-white/5 bg-[#0B0F19]">
                    <tr><td className="p-3 font-medium">Tiny</td><td className="p-3 text-blue-400">112K</td><td className="p-3">Off</td><td className="p-3">1</td></tr>
                    <tr><td className="p-3 font-medium">Small</td><td className="p-3 text-blue-400">4.6M</td><td className="p-3">Off</td><td className="p-3">1</td></tr>
                    <tr><td className="p-3 font-medium">Medium</td><td className="p-3 text-blue-400">34M</td><td className="p-3 text-green-400">On</td><td className="p-3">2</td></tr>
                    <tr><td className="p-3 font-medium">Large</td><td className="p-3 text-blue-400">102M</td><td className="p-3 text-green-400">On</td><td className="p-3">4</td></tr>
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      </section>

      
      {/* Data Pipeline */}
      <section className="py-24 px-6 relative">
        <div className="max-w-7xl mx-auto grid lg:grid-cols-2 gap-16 items-center">
          <div className="order-2 lg:order-1 bg-[#151C2C] p-2 rounded-3xl border border-white/5 shadow-2xl">
            <img src="/assets/diagrams/18-training-data-pipeline.png" alt="Training Data Pipeline" className="w-full rounded-2xl" />
          </div>
          <div className="order-1 lg:order-2">
            <h2 className="text-3xl md:text-5xl font-bold text-white mb-6">Data Pipeline</h2>
            <p className="text-lg text-slate-400 mb-6 leading-relaxed">
              We process real text corporas without overwhelming RAM. The exact same <code className="bg-slate-800 text-pink-400 px-1 py-0.5 rounded text-sm">(x, y, loss_mask)</code> triple pipeline is used for both synthetic benchmarks and actual text.
            </p>
            <ul className="space-y-4 text-sm text-slate-400 mb-8">
              <li>
                <strong className="text-white block mb-1">Dependency-Free Tokenization</strong>
                We use a <code className="text-blue-300">ByteTokenizer</code> (vocab=256) by default. It's dependency-free and provably round-trips any input, ensuring strict baseline comparability.
              </li>
              <li>
                <strong className="text-white block mb-1">Memmap-Backed Storage</strong>
                Memory tracks <code className="text-blue-300">batch_size * seq_len</code>, not corpus size. By using <code className="text-blue-300">np.memmap</code> on <code className="text-blue-300">train.bin</code>, a multi-GB corpus is never loaded into RAM wholesale.
              </li>
            </ul>
          </div>
        </div>
      </section>

      
      {/* Codebase Architecture */}
      <section id="codebase" className="py-24 px-6 relative">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-5xl font-bold text-white mb-6">Codebase Architecture</h2>
            <p className="text-lg text-slate-400 max-w-3xl mx-auto">
              Every file in this repository exists for one reason. Here is the full map.
            </p>
          </div>

          <div className="grid lg:grid-cols-3 gap-6 mb-16">
            <div className="bg-[#151C2C] border border-white/5 p-6 rounded-2xl">
              <h4 className="font-bold text-white mb-4 flex items-center gap-2">
                <span className="bg-blue-500/20 text-blue-400 px-2 py-1 rounded text-xs font-mono">layers/</span>
                Sequence Mixers
              </h4>
              <ul className="space-y-3 text-sm text-slate-400">
                <li><code className="text-blue-300">base.py</code> — The <code className="text-pink-400">SequenceMixer</code> contract. Every mixer implements <code className="text-white">forward(x)</code>, <code className="text-white">step(x_t, state)</code>, and <code className="text-white">init_state()</code>.</li>
                <li><code className="text-blue-300">s4d.py</code> — Diagonalized S4. Complex diagonal state, HiPPO-LegS init, fixed B.</li>
                <li><code className="text-blue-300">s5.py</code> — MIMO S5. Shared state, learned B and C matrices.</li>
                <li><code className="text-blue-300">registry.py</code> — Single source of truth. <code className="text-pink-400">get_mixer("s4d")</code> returns the class, defaults, paper citation, and family.</li>
              </ul>
            </div>
            <div className="bg-[#151C2C] border border-white/5 p-6 rounded-2xl">
              <h4 className="font-bold text-white mb-4 flex items-center gap-2">
                <span className="bg-purple-500/20 text-purple-400 px-2 py-1 rounded text-xs font-mono">models/</span>
                Backbone & Presets
              </h4>
              <ul className="space-y-3 text-sm text-slate-400">
                <li><code className="text-blue-300">backbone.py</code> — <code className="text-pink-400">SequenceBackbone</code>: embed → [norm → mixer → residual → norm → FFN → residual] × N → norm → head.</li>
                <li><code className="text-blue-300">presets.py</code> — Named scaling tiers (tiny/small/medium/large) bundling <code className="text-white">BackboneConfig</code> + <code className="text-white">TrainConfig</code> overrides with real param counts.</li>
              </ul>
            </div>
            <div className="bg-[#151C2C] border border-white/5 p-6 rounded-2xl">
              <h4 className="font-bold text-white mb-4 flex items-center gap-2">
                <span className="bg-green-500/20 text-green-400 px-2 py-1 rounded text-xs font-mono">utils/</span>
                Core Primitives
              </h4>
              <ul className="space-y-3 text-sm text-slate-400">
                <li><code className="text-blue-300">scan.py</code> — <code className="text-pink-400">parallel_scan_log</code> (Hillis-Steele) + <code className="text-pink-400">sequential_scan</code> (ground truth). Real + complex. Used by every model.</li>
                <li><code className="text-blue-300">hippo.py</code> — <code className="text-pink-400">hippo_diag_spectrum</code>. Extracted from S4D for reuse by S5 and all future diagonal models.</li>
              </ul>
            </div>
          </div>

          <div className="grid lg:grid-cols-3 gap-6">
            <div className="bg-[#151C2C] border border-white/5 p-6 rounded-2xl">
              <h4 className="font-bold text-white mb-4 flex items-center gap-2">
                <span className="bg-yellow-500/20 text-yellow-400 px-2 py-1 rounded text-xs font-mono">benchmarks/</span>
                Evaluation Suite
              </h4>
              <ul className="space-y-3 text-sm text-slate-400">
                <li><code className="text-blue-300">mqar.py</code> — Multi-Query Associative Recall (Zoology, Arora et al. 2024).</li>
                <li><code className="text-blue-300">copying.py</code> — Fixed-state capacity ceiling (Jelassi et al. 2024).</li>
                <li><code className="text-blue-300">induction_heads.py</code> — In-context pattern completion (Olsson et al. 2022).</li>
                <li><code className="text-blue-300">state_tracking.py</code> — Parity + S5 permutation (Merrill et al. 2024).</li>
                <li><code className="text-blue-300">lra/</code> — Legacy Long Range Arena (ListOps).</li>
              </ul>
            </div>
            <div className="bg-[#151C2C] border border-white/5 p-6 rounded-2xl">
              <h4 className="font-bold text-white mb-4 flex items-center gap-2">
                <span className="bg-red-500/20 text-red-400 px-2 py-1 rounded text-xs font-mono">training/</span>
                Trainer
              </h4>
              <ul className="space-y-3 text-sm text-slate-400">
                <li><code className="text-blue-300">trainer.py</code> — One trainer for all 14 models × 6 benchmarks. Masked cross-entropy, cosine LR schedule, mixed precision (bf16/fp16/fp32), gradient accumulation, checkpoint/resume.</li>
                <li>Every batch is the same <code className="text-pink-400">(x, y, loss_mask)</code> triple — swap <code className="text-white">train_batch_fn</code> and you've swapped benchmarks; swap <code className="text-white">model</code> and you've swapped architectures.</li>
              </ul>
            </div>
            <div className="bg-[#151C2C] border border-white/5 p-6 rounded-2xl">
              <h4 className="font-bold text-white mb-4 flex items-center gap-2">
                <span className="bg-cyan-500/20 text-cyan-400 px-2 py-1 rounded text-xs font-mono">data/</span>
                Real Text Pipeline
              </h4>
              <ul className="space-y-3 text-sm text-slate-400">
                <li><code className="text-blue-300">tokenizer.py</code> — <code className="text-pink-400">ByteTokenizer</code> (vocab=256, zero deps) + <code className="text-pink-400">GPT2Tokenizer</code> (via tiktoken, vocab=50257).</li>
                <li><code className="text-blue-300">text_lm.py</code> — <code className="text-pink-400">download_tiny_shakespeare()</code>, <code className="text-pink-400">prepare_corpus()</code> (to memmap .bin), <code className="text-pink-400">TextLMData.get_batch_fn()</code>.</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      
      {/* Quickstart */}
      <section id="quickstart" className="py-24 px-6 bg-[#0D1321] border-t border-white/5 relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-b from-blue-500/5 via-transparent to-purple-500/5 pointer-events-none"></div>
        <div className="max-w-4xl mx-auto relative z-10">
          <div className="text-center mb-12">
            <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-green-500/10 border border-green-500/30 text-green-400 text-sm font-bold tracking-wide mb-6">
              <Terminal className="w-4 h-4" />
              <span>Ready in Under a Minute</span>
            </div>
            <h2 className="text-3xl md:text-5xl font-bold text-white mb-6">Run the Verification Code</h2>
            <p className="text-lg text-slate-400 max-w-2xl mx-auto">
              You don't need a massive GPU cluster to verify the math. Prove parallel scan equivalence on a CPU.
            </p>
          </div>

          <div className="bg-black rounded-2xl p-6 md:p-8 font-mono text-sm shadow-[0_0_60px_rgba(0,0,0,0.5)] border border-white/10 relative group">
            <div className="absolute inset-0 rounded-2xl bg-gradient-to-b from-white/[0.03] to-transparent pointer-events-none"></div>
            <div className="flex gap-2 mb-6">
              <div className="w-3 h-3 rounded-full bg-red-500"></div>
              <div className="w-3 h-3 rounded-full bg-yellow-500"></div>
              <div className="w-3 h-3 rounded-full bg-green-500"></div>
              <span className="text-xs text-slate-600 ml-3 font-sans">Terminal — bash</span>
            </div>
            
            <p className="text-slate-500 mb-2"># 1. Clone the repository</p>
            <p className="text-white mb-6"><span className="text-green-400 select-none">$ </span>git clone https://github.com/SachinSSh/State-Space-Model.git<br/><span className="text-green-400 select-none">$ </span>cd State-Space-Model</p>
            
            <p className="text-slate-500 mb-2"># 2. Install dependencies</p>
            <p className="text-white mb-6"><span className="text-green-400 select-none">$ </span>pip install -e .<br/><span className="text-green-400 select-none">$ </span>pip install -r requirements-dev.txt</p>

            <p className="text-slate-500 mb-2"># 3. Run the verification test suite</p>
            <p className="text-green-400 mb-6"><span className="select-none">$ </span>pytest tests/ -v</p>

            <p className="text-slate-500 mb-2"># 4. Try the interactive notebooks</p>
            <p className="text-white"><span className="text-green-400 select-none">$ </span>jupyter notebook notebooks/00_setup_and_smoke_test.ipynb</p>
          </div>

          <div className="mt-12 text-center">
            <a href="https://github.com/SachinSSh/State-Space-Model" target="_blank" rel="noopener noreferrer" className="inline-flex items-center gap-3 bg-white text-slate-900 px-8 py-4 rounded-xl font-extrabold hover:bg-slate-100 hover:scale-105 transition-all shadow-[0_0_30px_rgba(255,255,255,0.2)]">
              <GitBranch className="w-5 h-5" />
              View on GitHub
              <ArrowRight className="w-4 h-4" />
            </a>
          </div>
        </div>
      </section>

      
    </>
  );
}
