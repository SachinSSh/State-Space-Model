
import { CheckCircle2, ShieldCheck } from 'lucide-react';
import { InlineMath, BlockMath } from 'react-katex';

export default function ModelsPage() {
  return (
    <>
      {/* Model Deep Dives */}
      <section id="models" className="py-24 px-6 bg-[#0D1321] border-y border-white/5">
        <div className="max-w-7xl mx-auto">
          <div className="mb-16 text-center">
            <h2 className="text-3xl md:text-5xl font-bold text-white mb-6">Model Deep Dives</h2>
            <p className="text-lg text-slate-400 max-w-2xl mx-auto">
              Real architectural analysis and verified benchmark results for the models shipped so far.
            </p>
          </div>

          <div className="space-y-8">
            
            {/* S4D Card */}
            <div className="bg-[#151C2C] border border-white/10 rounded-3xl overflow-hidden shadow-2xl">
              <div className="border-b border-white/5 bg-white/5 p-6 md:p-8 flex flex-wrap justify-between items-center gap-4">
                <div>
                  <div className="flex items-center gap-3 mb-1">
                    <h3 className="text-3xl font-extrabold text-white">S4D</h3>
                    <span className="bg-green-500/20 text-green-400 border border-green-500/30 px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider flex items-center gap-1.5">
                      <CheckCircle2 className="w-3.5 h-3.5" /> Shipped
                    </span>
                  </div>
                  <p className="text-slate-400">Foundation Model (Diagonalized S4)</p>
                </div>
              </div>

              <div className="p-6 md:p-8 grid lg:grid-cols-2 gap-12">
                <div>
                  <h4 className="text-lg font-bold text-white mb-4 border-b border-white/10 pb-2">The Recurrence (SISO)</h4>
                  <div className="bg-[#0B0F19] rounded-xl p-5 border border-white/5 text-center mb-6 overflow-x-auto shadow-inner">
                    <BlockMath math="\dot{x}(t) = A x(t) + B u(t), \quad y(t) = \mathrm{Re}(C x(t)) + D u(t)" />
                    <div className="my-3 border-b border-white/5 w-1/3 mx-auto"></div>
                    <BlockMath math="x_t = \bar{A} x_{t-1} + \bar{B} u_t, \quad y_t = \mathrm{Re}(C x_t) + D u_t" />
                  </div>
                  <ul className="space-y-4 text-sm text-slate-400">
                    <li><strong className="text-slate-200">Stable Decay:</strong> <InlineMath math="A = -e^{\text{log\_A\_real}} + i \cdot A_{\text{imag}}" /> ensures <InlineMath math="\mathrm{Re}(A) < 0" /> so gradients never explode.</li>
                    <li><strong className="text-slate-200">HiPPO Spectrum:</strong> Imaginary part is a diagonal approximation of the HiPPO-LegS spectrum.</li>
                    <li><strong className="text-slate-200">Fixed B:</strong> <InlineMath math="B" /> is all-ones, since magnitude is redundant with <InlineMath math="C" /> for a diagonal <InlineMath math="A" />.</li>
                  </ul>
                </div>
                
                <div>
                  <h4 className="text-lg font-bold text-white mb-4 border-b border-white/10 pb-2">Verifications & Benchmarks</h4>
                  <div className="space-y-3 mb-6">
                    <div className="flex items-start gap-3 text-sm text-slate-400">
                      <ShieldCheck className="text-blue-400 shrink-0 w-5 h-5" />
                      <p><strong>forward() == step():</strong> Parallel scan output exactly matches token-by-token recurrence to <code className="text-blue-300">1e-4</code> atol.</p>
                    </div>
                    <div className="flex items-start gap-3 text-sm text-slate-400">
                      <ShieldCheck className="text-blue-400 shrink-0 w-5 h-5" />
                      <p><strong>Causality Check:</strong> Perturbing input at <InlineMath math="t \ge 7" /> provably never alters output at <InlineMath math="t < 7" />.</p>
                    </div>
                  </div>
                  
                  <div className="bg-slate-900 rounded-xl p-5 border border-white/5 font-mono text-xs md:text-sm text-slate-300 space-y-3 shadow-inner">
                    <p className="text-slate-500 mb-2">// S4D's transition is a learned constant.</p>
                    <div><span className="text-green-400">Copying:</span> Holds up until copy_len ≈ d_state.</div>
                    <div><span className="text-red-400">MQAR:</span> 0.021 (Weak, near chance). Lacks routing.</div>
                    <div><span className="text-red-400">Induction Heads:</span> Near chance. Cannot retrieve.</div>
                    <div><span className="text-red-400">Parity / Tracking:</span> Weak to failing (Merrill 2024).</div>
                    <div className="mt-2 pt-2 border-t border-white/10"><span className="text-green-400">Real LM (Shakespeare):</span> 256 → <strong>11.4 PPL</strong> (69K params)</div>
                  </div>
                </div>
              </div>
            </div>

            {/* S5 Card */}
            <div className="bg-[#151C2C] border border-white/10 rounded-3xl overflow-hidden shadow-2xl">
              <div className="border-b border-white/5 bg-white/5 p-6 md:p-8 flex flex-wrap justify-between items-center gap-4">
                <div>
                  <div className="flex items-center gap-3 mb-1">
                    <h3 className="text-3xl font-extrabold text-white">S5</h3>
                    <span className="bg-green-500/20 text-green-400 border border-green-500/30 px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider flex items-center gap-1.5">
                      <CheckCircle2 className="w-3.5 h-3.5" /> Shipped
                    </span>
                  </div>
                  <p className="text-slate-400">Foundation Model (MIMO Channel Mixing)</p>
                </div>
              </div>

              <div className="p-6 md:p-8 grid lg:grid-cols-2 gap-12">
                <div>
                  <h4 className="text-lg font-bold text-white mb-4 border-b border-white/10 pb-2">The MIMO Difference</h4>
                  <p className="text-sm text-slate-400 mb-4 leading-relaxed">
                    S5 uses the exact same discrete-time recurrence equation as S4D, but changes the shapes. Instead of independent per-channel states, <InlineMath math="x_t \in \mathbb{C}^P" /> is a <strong>single state shared across every channel</strong>.
                  </p>
                  <div className="bg-[#0B0F19] rounded-xl p-5 border border-white/5 mb-6 text-sm text-slate-400 shadow-inner">
                    <p className="mb-2"><strong className="text-white">S4D (SISO):</strong> <InlineMath math="\bar{B}" /> and <InlineMath math="C" /> are scalars per channel.</p>
                    <p><strong className="text-white">S5 (MIMO):</strong> <InlineMath math="\bar{B} \in \mathbb{R}^{P \times d_{model}}" /> and <InlineMath math="C \in \mathbb{R}^{d_{model} \times P}" /> are true matrices mixing channels inside the recurrence.</p>
                  </div>
                  <ul className="space-y-4 text-sm text-slate-400">
                    <li><strong className="text-slate-200">Learned Matrices:</strong> Fixing <InlineMath math="B" /> to all-ones here would force all states to read identical combinations, destroying routing capacity. Both are fully learned.</li>
                    <li><strong className="text-slate-200">Selectivity Axis:</strong> S5 still uses a <em>learned constant</em> transition. The selectivity axis has NOT changed from S4D.</li>
                  </ul>
                </div>
                
                <div>
                  <h4 className="text-lg font-bold text-white mb-4 border-b border-white/10 pb-2">Verifications & Benchmarks</h4>
                  <div className="space-y-3 mb-6">
                    <div className="flex items-start gap-3 text-sm text-slate-400">
                      <ShieldCheck className="text-blue-400 shrink-0 w-5 h-5" />
                      <p><strong>State is Shared:</strong> Verified <code className="text-blue-300">init_state</code> returns <code className="text-blue-300">(batch, d_state)</code> with no channel dimension.</p>
                    </div>
                    <div className="flex items-start gap-3 text-sm text-slate-400">
                      <ShieldCheck className="text-blue-400 shrink-0 w-5 h-5" />
                      <p><strong>True MIMO Mixing:</strong> Perturbing a single input channel measurably affects <em>multiple</em> output channels directly.</p>
                    </div>
                  </div>
                  
                  <div className="bg-slate-900 rounded-xl p-5 border border-white/5 font-mono text-xs md:text-sm text-slate-300 space-y-3 shadow-inner">
                    <p className="text-slate-500 mb-2">// Proves MIMO doesn't fix recall without selectivity.</p>
                    <div><span className="text-green-400">Copying:</span> 0.959 acc (len 8, d_state 64). Holds up.</div>
                    <div><span className="text-red-400">MQAR:</span> 0.025 (Still near chance).</div>
                    <div><span className="text-red-400">Induction Heads:</span> 0.062 (At chance).</div>
                    <div><span className="text-red-400">S5 Group Tracking:</span> 0.111 (Weak).</div>
                    <div className="mt-2 pt-2 border-t border-white/10"><span className="text-green-400">Real LM (Shakespeare):</span> 256 → <strong>10.8 PPL</strong></div>
                  </div>
                </div>
              </div>
            </div>

            {/* Mamba (S6) Card */}
            <div className="bg-[#151C2C] border border-white/10 rounded-3xl overflow-hidden shadow-2xl">
              <div className="border-b border-white/5 bg-white/5 p-6 md:p-8 flex flex-wrap justify-between items-center gap-4">
                <div>
                  <div className="flex items-center gap-3 mb-1">
                    <h3 className="text-3xl font-extrabold text-white">Mamba (S6)</h3>
                    <span className="bg-blue-500/20 text-blue-400 border border-blue-500/30 px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider flex items-center gap-1.5">
                      Next Up
                    </span>
                  </div>
                  <p className="text-slate-400">Selective State Space Model</p>
                </div>
              </div>

              <div className="p-6 md:p-8 grid lg:grid-cols-2 gap-12">
                <div>
                  <h4 className="text-lg font-bold text-white mb-4 border-b border-white/10 pb-2">The Selective Mechanism</h4>
                  <p className="text-sm text-slate-400 mb-4 leading-relaxed">
                    Mamba abandons the learned constants of S4/S5. Instead, <InlineMath math="B" />, <InlineMath math="C" />, and the step size <InlineMath math="\Delta" /> become <strong>functions of the input token</strong>.
                  </p>
                  <ul className="space-y-4 text-sm text-slate-400">
                    <li><strong className="text-slate-200">Content-Aware Forgetting:</strong> The model can choose what to write and what to ignore by varying its step size dynamically.</li>
                    <li><strong className="text-slate-200">Loss of Convolution:</strong> Making the transition dynamic breaks the FFT convolution trick. Mamba relies on a hardware-aware fused scan.</li>
                  </ul>
                </div>
                
                <div>
                  <h4 className="text-lg font-bold text-white mb-4 border-b border-white/10 pb-2">Expected Benchmarks</h4>
                  <div className="bg-slate-900 rounded-xl p-5 border border-white/5 font-mono text-xs md:text-sm text-slate-300 space-y-3 shadow-inner">
                    <p className="text-slate-500 mb-2">// Projections based on literature.</p>
                    <div><span className="text-green-400">MQAR:</span> Expected ~0.9+ (Strong retrieval).</div>
                    <div><span className="text-green-400">Induction Heads:</span> Expected ~0.8+.</div>
                    <div><span className="text-yellow-400">Copying:</span> Cliff at state capacity.</div>
                    <div className="mt-2 pt-2 border-t border-white/10"><span className="text-slate-400">Real LM (Shakespeare):</span> Expected ~10 PPL</div>
                  </div>
                </div>
              </div>
            </div>

            {/* Mamba-2 (SSD) Card */}
            <div className="bg-[#151C2C] border border-white/10 rounded-3xl overflow-hidden shadow-2xl opacity-75">
              <div className="border-b border-white/5 bg-white/5 p-6 md:p-8 flex flex-wrap justify-between items-center gap-4">
                <div>
                  <div className="flex items-center gap-3 mb-1">
                    <h3 className="text-3xl font-extrabold text-white">Mamba-2 (SSD)</h3>
                    <span className="bg-slate-800 text-slate-400 border border-slate-700 px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider flex items-center gap-1.5">
                      Planned
                    </span>
                  </div>
                  <p className="text-slate-400">State Space Duality</p>
                </div>
              </div>

              <div className="p-6 md:p-8 grid lg:grid-cols-2 gap-12">
                <div>
                  <h4 className="text-lg font-bold text-white mb-4 border-b border-white/10 pb-2">The SSD Equivalence</h4>
                  <p className="text-sm text-slate-400 mb-4 leading-relaxed">
                    If you restrict the state matrix <InlineMath math="A" /> to a scalar times the identity (still input-dependent), it becomes mathematically equivalent to a masked form of linear attention.
                  </p>
                  <ul className="space-y-4 text-sm text-slate-400">
                    <li><strong className="text-slate-200">Tensor Core Chunking:</strong> Allows computation as chunked matrix multiplications on tensor cores for large speedups.</li>
                    <li><strong className="text-slate-200">Tradeoff:</strong> Reduced accuracy at fine-grained retrieval compared to Mamba-1's per-channel diagonal state.</li>
                  </ul>
                </div>
                
                <div>
                  <h4 className="text-lg font-bold text-white mb-4 border-b border-white/10 pb-2">Status</h4>
                  <div className="bg-slate-900 rounded-xl p-5 border border-white/5 font-mono text-xs md:text-sm text-slate-300 space-y-3 shadow-inner">
                    <p className="text-slate-500 mb-2">// Mamba-2 is currently in the planning phase.</p>
                    <p className="text-slate-400">We will verify SSD against linear attention primitives before integrating.</p>
                  </div>
                </div>
              </div>
            </div>

            {/* Mamba-3 Card */}
            <div className="bg-[#151C2C] border border-white/10 rounded-3xl overflow-hidden shadow-2xl opacity-75">
              <div className="border-b border-white/5 bg-white/5 p-6 md:p-8 flex flex-wrap justify-between items-center gap-4">
                <div>
                  <div className="flex items-center gap-3 mb-1">
                    <h3 className="text-3xl font-extrabold text-white">Mamba-3</h3>
                    <span className="bg-slate-800 text-slate-400 border border-slate-700 px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider flex items-center gap-1.5">
                      Planned
                    </span>
                  </div>
                  <p className="text-slate-400">Closing the Loop to HiPPO</p>
                </div>
              </div>

              <div className="p-6 md:p-8 grid lg:grid-cols-2 gap-12">
                <div>
                  <h4 className="text-lg font-bold text-white mb-4 border-b border-white/10 pb-2">The Complex State Returns</h4>
                  <p className="text-sm text-slate-400 mb-4 leading-relaxed">
                    Mamba-3 brings back the complex-valued state (from S4/S4D) specifically to aid with state-tracking, adds a MIMO formulation, and uses trapezoidal discretization.
                  </p>
                  <ul className="space-y-4 text-sm text-slate-400">
                    <li><strong className="text-slate-200">Overcoming TC⁰:</strong> Expected to perform significantly better on parity and permutation tracking compared to Mamba-1/2.</li>
                  </ul>
                </div>
                
                <div>
                  <h4 className="text-lg font-bold text-white mb-4 border-b border-white/10 pb-2">Status</h4>
                  <div className="bg-slate-900 rounded-xl p-5 border border-white/5 font-mono text-xs md:text-sm text-slate-300 space-y-3 shadow-inner">
                    <p className="text-slate-500 mb-2">// Mamba-3 is currently in the planning phase.</p>
                    <p className="text-slate-400">Will be integrated once reference code and detailed specifications are fully analyzed.</p>
                  </div>
                </div>
              </div>
            </div>

            {/* Linear Attention & Delta Rule */}
            <div className="bg-[#151C2C] border border-white/10 rounded-3xl overflow-hidden shadow-2xl opacity-75">
              <div className="border-b border-white/5 bg-white/5 p-6 md:p-8 flex flex-wrap justify-between items-center gap-4">
                <div>
                  <div className="flex items-center gap-3 mb-1">
                    <h3 className="text-3xl font-extrabold text-white">DeltaNet & GLA</h3>
                    <span className="bg-slate-800 text-slate-400 border border-slate-700 px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider flex items-center gap-1.5">
                      Planned
                    </span>
                  </div>
                  <p className="text-slate-400">The Linear Attention Sub-family</p>
                </div>
              </div>

              <div className="p-6 md:p-8 grid lg:grid-cols-2 gap-12">
                <div>
                  <h4 className="text-lg font-bold text-white mb-4 border-b border-white/10 pb-2">Online Correction</h4>
                  <p className="text-sm text-slate-400 mb-4 leading-relaxed">
                    Reframes the state update as an online gradient-descent or <strong>delta-rule</strong> correction. Instead of only adding new key-value associations, it can overwrite what's already stored for a given key.
                  </p>
                  <ul className="space-y-4 text-sm text-slate-400">
                    <li><strong className="text-slate-200">Gated Linear Attention (GLA):</strong> Adds a data-dependent forget gate on top of linear attention.</li>
                    <li><strong className="text-slate-200">DeltaNet:</strong> Further introduces the ability to overwrite memory using error-driven updates.</li>
                  </ul>
                </div>
                
                <div>
                  <h4 className="text-lg font-bold text-white mb-4 border-b border-white/10 pb-2">Status</h4>
                  <div className="bg-slate-900 rounded-xl p-5 border border-white/5 font-mono text-xs md:text-sm text-slate-300 space-y-3 shadow-inner">
                    <p className="text-slate-500 mb-2">// Linear attention models are planned next.</p>
                    <p className="text-slate-400">Will test their associative recall capacities against Mamba's diagonal state approach.</p>
                  </div>
                </div>
              </div>
            </div>

            {/* Hybrids Card */}
            <div className="bg-[#151C2C] border border-white/10 rounded-3xl overflow-hidden shadow-2xl opacity-75">
              <div className="border-b border-white/5 bg-white/5 p-6 md:p-8 flex flex-wrap justify-between items-center gap-4">
                <div>
                  <div className="flex items-center gap-3 mb-1">
                    <h3 className="text-3xl font-extrabold text-white">Hybrids (Jamba, Griffin)</h3>
                    <span className="bg-slate-800 text-slate-400 border border-slate-700 px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider flex items-center gap-1.5">
                      Planned
                    </span>
                  </div>
                  <p className="text-slate-400">Solving the Capacity Ceiling</p>
                </div>
              </div>

              <div className="p-6 md:p-8 grid lg:grid-cols-2 gap-12">
                <div>
                  <h4 className="text-lg font-bold text-white mb-4 border-b border-white/10 pb-2">Why pure SSMs aren't enough</h4>
                  <p className="text-sm text-slate-400 mb-4 leading-relaxed">
                    Any model with a state size independent of sequence length <strong>cannot copy an arbitrary string</strong> once it exceeds what the fixed state can hold. Attention's "state" is the growing KV cache, so it doesn't face this ceiling.
                  </p>
                  <ul className="space-y-4 text-sm text-slate-400">
                    <li><strong className="text-slate-200">The 10% Solution:</strong> Hybrids interleave Transformer layers with SSM layers at roughly a 1:7 ratio.</li>
                    <li><strong className="text-slate-200">Best of both worlds:</strong> O(1) inference for most layers, with periodic KV caches to preserve exact recall and in-context learning.</li>
                  </ul>
                </div>
                
                <div>
                  <h4 className="text-lg font-bold text-white mb-4 border-b border-white/10 pb-2">Status</h4>
                  <div className="bg-slate-900 rounded-xl p-5 border border-white/5 font-mono text-xs md:text-sm text-slate-300 space-y-3 shadow-inner">
                    <p className="text-slate-500 mb-2">// Architectural proofs planned.</p>
                    <p className="text-slate-400">We will verify that hybridizing the stack strictly pushes the copying capacity ceiling further back.</p>
                  </div>
                </div>
              </div>
            </div>

            {/* RWKV-7 Card */}
            <div className="bg-[#151C2C] border border-white/10 rounded-3xl overflow-hidden shadow-2xl opacity-75">
              <div className="border-b border-white/5 bg-white/5 p-6 md:p-8 flex flex-wrap justify-between items-center gap-4">
                <div>
                  <div className="flex items-center gap-3 mb-1">
                    <h3 className="text-3xl font-extrabold text-white">RWKV-7 "Goose"</h3>
                    <span className="bg-slate-800 text-slate-400 border border-slate-700 px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider flex items-center gap-1.5">
                      Planned
                    </span>
                  </div>
                  <p className="text-slate-400">Dynamic State Evolution</p>
                </div>
              </div>

              <div className="p-6 md:p-8 grid lg:grid-cols-2 gap-12">
                <div>
                  <h4 className="text-lg font-bold text-white mb-4 border-b border-white/10 pb-2">Expressive Memory</h4>
                  <p className="text-sm text-slate-400 mb-4 leading-relaxed">
                    RWKV-7 pushes the linear attention paradigm forward by introducing vector-valued gating and per-step learning rates.
                  </p>
                  <ul className="space-y-4 text-sm text-slate-400">
                    <li><strong className="text-slate-200">Formal Guarantees:</strong> Provably capable of performing state tracking and recognizing all regular languages—tasks where plain linear attention fails.</li>
                    <li><strong className="text-slate-200">Parallel Training:</strong> Maintains the crucial ability to be trained in parallel like a Transformer while offering an RNN-like fast inference mode.</li>
                  </ul>
                </div>
                
                <div>
                  <h4 className="text-lg font-bold text-white mb-4 border-b border-white/10 pb-2">Status</h4>
                  <div className="bg-slate-900 rounded-xl p-5 border border-white/5 font-mono text-xs md:text-sm text-slate-300 space-y-3 shadow-inner">
                    <p className="text-slate-500 mb-2">// Added to the linear-attention roadmap.</p>
                    <p className="text-slate-400">Will be evaluated explicitly on the state tracking (NC¹ permutation) benchmark.</p>
                  </div>
                </div>
              </div>
            </div>

            {/* xLSTM Card */}
            <div className="bg-[#151C2C] border border-white/10 rounded-3xl overflow-hidden shadow-2xl opacity-75">
              <div className="border-b border-white/5 bg-white/5 p-6 md:p-8 flex flex-wrap justify-between items-center gap-4">
                <div>
                  <div className="flex items-center gap-3 mb-1">
                    <h3 className="text-3xl font-extrabold text-white">xLSTM (mLSTM)</h3>
                    <span className="bg-slate-800 text-slate-400 border border-slate-700 px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider flex items-center gap-1.5">
                      Planned
                    </span>
                  </div>
                  <p className="text-slate-400">The Modernized LSTM</p>
                </div>
              </div>

              <div className="p-6 md:p-8 grid lg:grid-cols-2 gap-12">
                <div>
                  <h4 className="text-lg font-bold text-white mb-4 border-b border-white/10 pb-2">Returning to Roots</h4>
                  <p className="text-sm text-slate-400 mb-4 leading-relaxed">
                    Rather than starting from attention and removing softmax, xLSTM starts from the classic LSTM and modernizes it with exponential gating and matrix-valued memory structures.
                  </p>
                  <ul className="space-y-4 text-sm text-slate-400">
                    <li><strong className="text-slate-200">Matrix Memory (mLSTM):</strong> Upgrades the scalar memory cell to a matrix, vastly increasing the capacity to store associations.</li>
                    <li><strong className="text-slate-200">Pareto-Dominant:</strong> Recent scaling laws demonstrate it outperforms Transformers at matched training compute.</li>
                  </ul>
                </div>
                
                <div>
                  <h4 className="text-lg font-bold text-white mb-4 border-b border-white/10 pb-2">Status</h4>
                  <div className="bg-slate-900 rounded-xl p-5 border border-white/5 font-mono text-xs md:text-sm text-slate-300 space-y-3 shadow-inner">
                    <p className="text-slate-500 mb-2">// Architecture review in progress.</p>
                    <p className="text-slate-400">Planned for integration into the standard sequence mixer pipeline for head-to-head evaluation.</p>
                  </div>
                </div>
              </div>
            </div>

            {/* Titans & Vision Mamba */}
            <div className="grid md:grid-cols-2 gap-8">
              <div className="bg-[#151C2C] border border-white/10 rounded-3xl overflow-hidden shadow-2xl opacity-75 flex flex-col">
                <div className="border-b border-white/5 bg-white/5 p-6 flex-wrap justify-between items-center gap-4">
                  <div className="flex items-center gap-3 mb-1">
                    <h3 className="text-2xl font-extrabold text-white">Titans (mini)</h3>
                    <span className="bg-slate-800 text-slate-400 border border-slate-700 px-2 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-wider">Planned</span>
                  </div>
                  <p className="text-slate-400 text-sm">Learning to Memorize at Test Time</p>
                </div>
                <div className="p-6 flex-grow">
                  <p className="text-sm text-slate-400 mb-4 leading-relaxed">
                    Introduces a neural-memory module that dynamically learns what to memorize from the context <em>at test time</em>, combining attention for short-term memory and neural updates for the long term.
                  </p>
                </div>
              </div>

              <div className="bg-[#151C2C] border border-white/10 rounded-3xl overflow-hidden shadow-2xl opacity-75 flex flex-col">
                <div className="border-b border-white/5 bg-white/5 p-6 flex-wrap justify-between items-center gap-4">
                  <div className="flex items-center gap-3 mb-1">
                    <h3 className="text-2xl font-extrabold text-white">Vision Mamba</h3>
                    <span className="bg-slate-800 text-slate-400 border border-slate-700 px-2 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-wider">Planned</span>
                  </div>
                  <p className="text-slate-400 text-sm">Bidirectional Applied SSMs</p>
                </div>
                <div className="p-6 flex-grow">
                  <p className="text-sm text-slate-400 mb-4 leading-relaxed">
                    Adapts the selective scan by making it <strong>bidirectional</strong>, replacing Vision Transformers (ViTs). It demonstrates higher efficiency and accuracy on dense prediction tasks.
                  </p>
                </div>
              </div>
            </div>

          </div>
        </div>
      </section>

      
      {/* The 14-Model Roadmap */}
      <section className="py-24 px-6 bg-[#0D1321] border-y border-white/5">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-5xl font-bold text-white mb-6">The 14-Model Roadmap</h2>
            <p className="text-lg text-slate-400 max-w-3xl mx-auto">
              Every model is a different <code className="bg-slate-800 text-pink-400 px-1 py-0.5 rounded text-sm">SequenceMixer</code> dropped into the exact same stack. 
            </p>
          </div>

          {/* Research Lineage Timeline */}
          <div className="mb-20 bg-[#151C2C] border border-white/5 p-8 md:p-12 rounded-3xl">
            <h3 className="text-2xl font-bold text-white mb-8 border-b border-white/5 pb-4">A Brief History (2020 - 2026)</h3>
            <div className="grid md:grid-cols-2 gap-12">
              <div>
                <h4 className="text-lg font-bold text-blue-400 mb-3">1. S4/S4D: The Foundations (2022)</h4>
                <p className="text-sm text-slate-400 mb-6 leading-relaxed">
                  Building on the HiPPO framework (2020), S4 and S4D introduced practical linear SSMs. <strong>A, B, C are learned constants.</strong> The model controls how fast to forget, but not what to remember.
                </p>
                <h4 className="text-lg font-bold text-blue-400 mb-3">2. The Mamba Lineage (2023 - 2026)</h4>
                <p className="text-sm text-slate-400 mb-2 leading-relaxed">
                  <strong className="text-white">Mamba (S6)</strong> makes A, B, C functions of the input token. It can <em>select</em> what to write into state.
                </p>
                <p className="text-sm text-slate-400 mb-2 leading-relaxed">
                  <strong className="text-white">Mamba-2 (SSD)</strong> proved State Space Duality: a restricted recurrence is mathematically identical to masked linear attention, enabling tensor core chunking.
                </p>
                <p className="text-sm text-slate-400 leading-relaxed">
                  <strong className="text-white">Mamba-3</strong> adds trapezoidal discretization and complex-valued states for better state-tracking.
                </p>
              </div>
              <div>
                <h4 className="text-lg font-bold text-purple-400 mb-3">3. Linear Attention & Delta Rule</h4>
                <p className="text-sm text-slate-400 mb-6 leading-relaxed">
                  <strong>RetNet</strong> applied fixed multi-scale decay. <strong>GLA</strong> added a data-dependent forget gate. <strong>DeltaNet</strong> reframed state update as online gradient-descent correction. <strong>RWKV-7</strong> generalized vector-valued gating. <strong>xLSTM</strong> modernized the LSTM with matrix-valued memory.
                </p>
                <h4 className="text-lg font-bold text-purple-400 mb-3">4. Hybrids</h4>
                <p className="text-sm text-slate-400 leading-relaxed">
                  Models like <strong>Jamba</strong>, <strong>Nemotron-H</strong>, and <strong>Griffin</strong> keep 5-15% real softmax attention to solve the provable capacity ceiling (Jelassi et al. 2024), where pure SSMs fail to copy arbitrary long strings.
                </p>
              </div>
            </div>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4 mb-20">
            {[
              { id: 1, name: "S4D", family: "foundation", status: "Done" },
              { id: 2, name: "S5", family: "foundation", status: "Done" },
              { id: 3, name: "Mamba / S6", family: "mamba lineage", status: "Next Up" },
              { id: 4, name: "Mamba-2 / SSD", family: "mamba lineage", status: "Planned" },
              { id: 5, name: "Mamba-3", family: "mamba lineage", status: "Planned" },
              { id: 6, name: "RetNet", family: "linear-attn", status: "Planned" },
              { id: 7, name: "GLA", family: "linear-attn", status: "Planned" },
              { id: 8, name: "DeltaNet", family: "linear-attn", status: "Planned" },
              { id: 9, name: "RWKV-7 'Goose'", family: "linear-attn", status: "Planned" },
              { id: 10, name: "xLSTM (mLSTM)", family: "linear-attn", status: "Planned" },
              { id: 11, name: "Jamba-style block", family: "hybrid", status: "Planned" },
              { id: 12, name: "Griffin", family: "hybrid", status: "Planned" },
              { id: 13, name: "Vision Mamba", family: "applied", status: "Planned" },
              { id: 14, name: "Titans (mini)", family: "memory frontier", status: "Planned" }
            ].map((model) => (
              <div key={model.id} className="bg-[#151C2C] border border-white/5 p-5 rounded-xl hover:border-white/20 transition-colors">
                <div className="flex justify-between items-start mb-2">
                  <h4 className="font-bold text-white flex items-center gap-2">
                    <span className="text-slate-500 text-xs">{model.id}.</span> {model.name}
                  </h4>
                  <span className={`text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded-full ${
                    model.status === 'Done' ? 'bg-green-500/20 text-green-400' :
                    model.status === 'Next Up' ? 'bg-blue-500/20 text-blue-400' :
                    'bg-slate-800 text-slate-400'
                  }`}>
                    {model.status}
                  </span>
                </div>
                <p className="text-xs text-slate-500 font-mono uppercase tracking-widest">{model.family}</p>
              </div>
            ))}
          </div>
          
          <div className="grid lg:grid-cols-2 gap-12 items-center">
            <div>
              <h3 className="text-2xl font-bold text-white mb-6">Verification Methodology</h3>
              <p className="text-slate-400 mb-6 leading-relaxed">
                To be added to this repository, a model must implement a test verifying its scan-based <code className="text-blue-300">forward()</code> exactly matches an independent <code className="text-blue-300">step()</code> computation to <code className="text-white">1e-4</code> atol.
              </p>
              <p className="text-slate-400 leading-relaxed">
                We don't just check "did it run without error" — we explicitly verify the discretization math using <code className="bg-slate-800 text-pink-400 px-1 py-0.5 rounded text-sm">torch.allclose</code> mathematically before accepting a model.
              </p>
            </div>
            <div className="bg-[#151C2C] p-2 rounded-2xl border border-white/5 shadow-2xl">
              <img src="/assets/diagrams/12-verification-methodology.png" alt="Verification Methodology Flowchart" className="w-full rounded-xl" />
            </div>
          </div>

        </div>
      </section>

      
    </>
  );
}
