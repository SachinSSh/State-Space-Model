


export default function BenchmarksPage() {
  return (
    <>
      {/* Benchmarks */}
      <section id="benchmarks" className="py-24 px-6 bg-[#0D1321] border-y border-white/5">
        <div className="max-w-7xl mx-auto">
          <div className="mb-16 text-center">
            <h2 className="text-3xl md:text-5xl font-bold text-white mb-6">Benchmarks & Structural Limits</h2>
            <p className="text-lg text-slate-400 max-w-2xl mx-auto">
              We evaluate every model against six strict tasks specifically chosen to expose theoretical limits, rather than hyperparameter luck.
            </p>
          </div>

          <div className="grid lg:grid-cols-2 gap-16 items-center mb-20">
            <div>
              <h3 className="text-2xl font-bold text-white mb-6">The TC⁰ Ceiling</h3>
              <p className="text-lg text-slate-400 mb-6 leading-relaxed">
                Merrill, Petty & Sabharwal (2024) proved that any model whose state transition is a <strong>diagonal, decay-only linear map</strong> is restricted to computations in TC⁰ — the same class as plain softmax attention.
              </p>
              <p className="text-slate-400 mb-6 leading-relaxed">
                S5 permutation composition is NC¹-complete. This is the standard benchmark for demonstrating this limitation in practice. Parity (also in our suite) is a gentler TC⁰ sanity check.
              </p>
              <div className="bg-[#151C2C] border border-white/5 p-6 rounded-2xl">
                <h4 className="text-white font-bold mb-3">Why This Matters for the Roadmap</h4>
                <p className="text-sm text-slate-400">
                  DeltaNet's delta-rule correction, RWKV-7's generalized delta rule, and Mamba-3's complex-valued state are <em>each</em> designed to close this gap. Our benchmark suite is specifically built to track whether they actually do.
                </p>
              </div>
            </div>
            <div className="bg-[#151C2C] p-2 rounded-2xl border border-white/5 shadow-2xl">
              <img src="/assets/diagrams/05-selective-vs-nonselective.png" alt="Selective vs Non-selective" className="w-full rounded-xl" />
            </div>
          </div>

          {/* Comparative Results Table */}
          <div>
            <h3 className="text-2xl font-bold text-white mb-8 text-center">Comparative Benchmark Results</h3>
            <div className="overflow-x-auto rounded-2xl border border-white/10">
              <table className="w-full text-left text-sm text-slate-300">
                <thead className="bg-white/5 border-b border-white/10">
                  <tr>
                    <th className="p-4 font-bold text-white">Benchmark</th>
                    <th className="p-4 font-bold text-white">What It Tests</th>
                    <th className="p-4 font-bold text-white text-center">S4D</th>
                    <th className="p-4 font-bold text-white text-center">S5</th>
                    <th className="p-4 font-bold text-white text-center">Mamba (expected)</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-white/5 bg-[#0B0F19]">
                  <tr>
                    <td className="p-4 font-medium">MQAR</td>
                    <td className="p-4 text-slate-400">Content-based associative recall</td>
                    <td className="p-4 text-center text-red-400 font-mono">0.021</td>
                    <td className="p-4 text-center text-red-400 font-mono">0.025</td>
                    <td className="p-4 text-center text-green-400 font-mono">~0.9+</td>
                  </tr>
                  <tr>
                    <td className="p-4 font-medium">Copying</td>
                    <td className="p-4 text-slate-400">Fixed-state capacity ceiling</td>
                    <td className="p-4 text-center text-yellow-400 font-mono">cliff @ d_state</td>
                    <td className="p-4 text-center text-green-400 font-mono">0.959</td>
                    <td className="p-4 text-center text-yellow-400 font-mono">cliff @ d_state</td>
                  </tr>
                  <tr>
                    <td className="p-4 font-medium">Induction Heads</td>
                    <td className="p-4 text-slate-400">In-context pattern completion</td>
                    <td className="p-4 text-center text-red-400 font-mono">chance</td>
                    <td className="p-4 text-center text-red-400 font-mono">0.062</td>
                    <td className="p-4 text-center text-green-400 font-mono">~0.8+</td>
                  </tr>
                  <tr>
                    <td className="p-4 font-medium">Parity (TC⁰)</td>
                    <td className="p-4 text-slate-400">Diagonal decay expressivity</td>
                    <td className="p-4 text-center text-red-400 font-mono">failing</td>
                    <td className="p-4 text-center text-red-400 font-mono">weak</td>
                    <td className="p-4 text-center text-yellow-400 font-mono">TBD</td>
                  </tr>
                  <tr>
                    <td className="p-4 font-medium">S5 Tracking</td>
                    <td className="p-4 text-slate-400">NC¹ permutation composition</td>
                    <td className="p-4 text-center text-slate-500 font-mono">N/A</td>
                    <td className="p-4 text-center text-red-400 font-mono">0.111</td>
                    <td className="p-4 text-center text-yellow-400 font-mono">TBD</td>
                  </tr>
                  <tr>
                    <td className="p-4 font-medium">Real LM (Shakespeare)</td>
                    <td className="p-4 text-slate-400">Byte-level perplexity</td>
                    <td className="p-4 text-center text-green-400 font-mono">11.4 PPL</td>
                    <td className="p-4 text-center text-green-400 font-mono">10.8 PPL</td>
                    <td className="p-4 text-center text-green-400 font-mono">~10 PPL</td>
                  </tr>
                </tbody>
              </table>
            </div>
            <p className="text-xs text-slate-500 mt-4 text-center">
              S4D/S5 values are from verified runs in this repo. Mamba values are projections based on the published literature, to be filled in when Model 3 ships.
            </p>
          </div>
        </div>
      </section>

      
      {/* Test Suite */}
      <section className="py-24 px-6 relative">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-5xl font-bold text-white mb-6">13 Test Files. Zero Trust.</h2>
            <p className="text-lg text-slate-400 max-w-3xl mx-auto">
              Every claim in this repository is backed by a runnable test. Here is the full verification surface.
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
            {[
              { file: "test_scan.py", desc: "parallel_scan_log matches sequential_scan to 1e-4 for real and complex dtypes." },
              { file: "test_s4d.py", desc: "Output shape, forward == step, causality check, complex gradient finite, optimizer step." },
              { file: "test_s5.py", desc: "MIMO shape, forward == step, B and C actually mix channels, shared state has no channel dim." },
              { file: "test_backbone.py", desc: "End-to-end backbone shape, gradient flow, weight tying verification." },
              { file: "test_benchmarks.py", desc: "MQAR, copying, induction heads, state tracking data generators produce correct shapes and masks." },
              { file: "test_checkpoint.py", desc: "Reload produces bit-identical eval loss. Optimizer state + RNG streams restored." },
              { file: "test_grad_checkpointing.py", desc: "Identical loss and gradients with checkpointing on vs off. OOM config succeeds with it on." },
              { file: "test_grad_accum.py", desc: "k micro-batches match one k×-larger batch to ~1e-8 numerical agreement." },
              { file: "test_text_lm.py", desc: "Full pipeline: download → tokenize → prepare_corpus → TextLMData → get_batch_fn → train." },
              { file: "test_presets.py", desc: "Named presets produce correct param counts. get_preset() returns valid configs." },
              { file: "test_export_for_inference.py", desc: "Exported weights load into ssm-infer. Forward pass parity to 1e-4 max diff." },
              { file: "test_listops.py", desc: "LRA ListOps synthetic generation, shape verification." },
            ].map((t) => (
              <div key={t.file} className="bg-[#151C2C] border border-white/5 p-5 rounded-xl hover:border-white/20 transition-colors">
                <h4 className="font-bold text-blue-400 font-mono text-sm mb-2">{t.file}</h4>
                <p className="text-xs text-slate-400 leading-relaxed">{t.desc}</p>
              </div>
            ))}
          </div>

          <div className="mt-12 bg-black rounded-2xl p-6 border border-white/10 max-w-2xl mx-auto">
            <div className="flex gap-2 mb-4">
              <div className="w-3 h-3 rounded-full bg-red-500"></div>
              <div className="w-3 h-3 rounded-full bg-yellow-500"></div>
              <div className="w-3 h-3 rounded-full bg-green-500"></div>
            </div>
            <p className="text-slate-500 font-mono text-sm mb-1"># Run the entire verification suite</p>
            <p className="text-green-400 font-mono text-sm mb-4">pytest tests/ -v --tb=short</p>
            <p className="text-slate-500 font-mono text-sm mb-1"># Run just the mathematical proofs</p>
            <p className="text-green-400 font-mono text-sm">pytest tests/test_scan.py tests/test_s4d.py tests/test_s5.py -v</p>
          </div>
        </div>
      </section>

      
      {/* Notebooks */}
      <section className="py-24 px-6 bg-[#0D1321] border-y border-white/5">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-5xl font-bold text-white mb-6">Interactive Notebooks</h2>
            <p className="text-lg text-slate-400 max-w-3xl mx-auto">
              Every model ships with a runnable Jupyter notebook that trains, evaluates, and visualizes results end-to-end.
            </p>
          </div>

          <div className="grid md:grid-cols-3 gap-8">
            <div className="bg-[#151C2C] border border-white/5 rounded-2xl overflow-hidden group hover:border-white/20 transition-colors">
              <div className="bg-gradient-to-r from-blue-600/20 to-purple-600/20 p-6 border-b border-white/5">
                <h4 className="text-xl font-bold text-white">00_setup_and_smoke_test</h4>
                <p className="text-sm text-slate-400 mt-1">Environment & Smoke Test</p>
              </div>
              <div className="p-6 text-sm text-slate-400 space-y-2">
                <p>Verifies imports, CUDA availability, and runs a minimal forward pass through the backbone.</p>
                <p>Lists all registered mixers from the registry. Confirms the shared infrastructure works before any model-specific work.</p>
              </div>
            </div>
            <div className="bg-[#151C2C] border border-white/5 rounded-2xl overflow-hidden group hover:border-white/20 transition-colors">
              <div className="bg-gradient-to-r from-green-600/20 to-cyan-600/20 p-6 border-b border-white/5">
                <h4 className="text-xl font-bold text-white">01_s4.ipynb</h4>
                <p className="text-sm text-slate-400 mt-1">S4D Full Evaluation</p>
              </div>
              <div className="p-6 text-sm text-slate-400 space-y-2">
                <p>Trains S4D on all 6 benchmarks: MQAR, Copying, Induction Heads, Parity/Tracking, LRA ListOps, and real Tiny-Shakespeare LM.</p>
                <p>Includes throughput-vs-length scaling analysis and gradient checkpointing demo. Saves results to <code className="text-blue-300">results/s4d_results.json</code>.</p>
              </div>
            </div>
            <div className="bg-[#151C2C] border border-white/5 rounded-2xl overflow-hidden group hover:border-white/20 transition-colors">
              <div className="bg-gradient-to-r from-orange-600/20 to-red-600/20 p-6 border-b border-white/5">
                <h4 className="text-xl font-bold text-white">02_s5.ipynb</h4>
                <p className="text-sm text-slate-400 mt-1">S5 Full Evaluation</p>
              </div>
              <div className="p-6 text-sm text-slate-400 space-y-2">
                <p>Same structure as 01_s4. Trains S5 on all 6 benchmarks with <code className="text-blue-300">FAST_DEV_RUN</code> flag for quick CPU verification.</p>
                <p>Demonstrates the MIMO advantage over SISO on copying. Saves to <code className="text-blue-300">results/s5_results.json</code> for cross-model comparison.</p>
              </div>
            </div>
          </div>
        </div>
      </section>

      
      {/* References / Papers */}
      <section className="py-24 px-6 relative">
        <div className="max-w-5xl mx-auto">
          <h2 className="text-3xl md:text-5xl font-bold text-white mb-12 text-center">References</h2>
          <div className="grid md:grid-cols-2 gap-x-12 gap-y-4 text-sm text-slate-400">
            {[
              "Gu, Dao, Ermon, Rudra & Ré. HiPPO: Recurrent Memory with Optimal Polynomial Projections. NeurIPS 2020.",
              "Gu, Goel & Ré. Efficiently Modeling Long Sequences with Structured State Spaces. ICLR 2022. (S4)",
              "Gu, Gupta, Goel & Ré. On the Parameterization and Initialization of Diagonal State Space Models. NeurIPS 2022. (S4D)",
              "Smith, Warrington & Linderman. Simplified State Space Layers for Sequence Modeling. ICLR 2023. (S5)",
              "Gu & Dao. Mamba: Linear-Time Sequence Modeling with Selective State Spaces. 2023.",
              "Dao & Gu. Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality. 2024. (Mamba-2)",
              "Gu, Dao, Kolter et al. Mamba-3. ICLR 2026.",
              "Sun et al. Retentive Network: A Successor to Transformer for Large Language Models. 2023. (RetNet)",
              "Yang, Wang, Shen, Panda & Kim. Gated Linear Attention Transformers with Hardware-Efficient Training. 2024. (GLA)",
              "Yang, Wang, Zhang, Shen & Kim. Parallelizing Linear Transformers with the Delta Rule. 2024. (DeltaNet)",
              "Peng et al. RWKV-7 \"Goose\" with Expressive Dynamic State Evolution. 2025.",
              "Beck et al. xLSTM: Extended Long Short-Term Memory. 2024.",
              "Lieber et al. (AI21). Jamba: A Hybrid Transformer-Mamba Language Model. 2024.",
              "De, Smith, Fernando et al. (Google DeepMind). Griffin: Mixing Gated Linear Recurrences with Local Attention. 2024.",
              "Behrouz et al. (Google). Titans: Learning to Memorize at Test Time. 2024/25.",
              "Jelassi, Brandfonbrener, Kakade & Malach. Repeat After Me: Transformers are Better than SSMs at Copying. 2024.",
              "Merrill, Petty & Sabharwal. The Illusion of State in State-Space Models. 2024.",
              "Arora, Eyuboglu, Zhang et al. Zoology: Measuring and Improving Recall in Efficient Language Models. 2024.",
              "Olsson et al. (Anthropic). In-context Learning and Induction Heads. 2022.",
              "Tay et al. Long Range Arena: A Benchmark for Efficient Transformers. 2020.",
            ].map((ref, i) => (
              <p key={i} className="py-2 border-b border-white/5 leading-relaxed">
                <span className="text-slate-600 font-mono text-xs mr-2">[{i + 1}]</span>
                {ref}
              </p>
            ))}
          </div>
        </div>
      </section>

      
    </>
  );
}
