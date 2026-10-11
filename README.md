# ⚡ On-Device LLM Benchmarking: Quantized Qwen2.5-Coder 1.5B vs. 7B

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Benchmark](https://img.shields.io/badge/Benchmark-HumanEval%20%7C%20EvalPlus-brightgreen)
![Models](https://img.shields.io/badge/Models-Qwen2.5--Coder%20(Q4__K__M)-purple)
![Hardware](https://img.shields.io/badge/GPU-NVIDIA%20RTX%204050%20(6GB%20VRAM)-76B900?logo=nvidia&logoColor=white)

A comprehensive empirical evaluation and automated benchmarking pipeline comparing **4-bit quantized (`Q4_K_M` GGUF)** code-generation Large Language Models—**`Qwen2.5-Coder-1.5B-Instruct`** and **`Qwen2.5-Coder-7B-Instruct`**—deployed locally on consumer-grade hardware via **LM Studio**.

Developed for **ECE241 — Computer Systems Engineering** at the **School of Engineering and Applied Sciences, Nile University**.

📄 **[Read the Full Technical Report (PDF)](./docs/ECE241_Project_Report.pdf)**

---

## 🎯 Project Overview & Objectives

While cloud-hosted LLM inference introduces network latency, privacy risks, and recurring API costs, post-training 4-bit quantization (`Q4_K_M`) enables billion-parameter code models to run entirely offline on consumer laptops. This repository provides a **4-layer automated Python orchestration & telemetry framework** (`benchmark.py`) designed to:

1. **Evaluate Code Synthesis Accuracy:** Benchmark `Qwen2.5-Coder 1.5B` vs. `7B` across all 164 Python problems in the **HumanEval** and stricter **HumanEval+ (EvalPlus)** test suites under deterministic greedy decoding (`temperature = 0`).
2. **Profile Hardware Efficiency:** Asynchronously monitor real-time **GPU VRAM (`nvidia-smi`)**, **System RAM (`psutil`)**, **Inference Throughput (`tokens/sec`)**, and **Time to First Token (`TTFT`)** without blocking the inference path.
3. **Ensure Memory Safety on 6 GB VRAM:** Automate sequential Just-In-Time (JIT) model loading and unloading via LM Studio's REST API v1 to prevent Out-Of-Memory (OOM) faults.

---

## 📊 Key Experimental Results

All benchmarks were executed under identical deterministic conditions (`Q4_K_M` quantization, `4096` context window, `512` max new tokens, `temperature = 0`) on an **NVIDIA GeForce RTX 4050 (6 GB VRAM) / 16 GB DDR5 RAM** system:

| Metric | Qwen2.5-Coder 1.5B (`Q4_K_M`) | Qwen2.5-Coder 7B (`Q4_K_M`) | Winner | Delta |
| :--- | :---: | :---: | :---: | :---: |
| **Pass@1 — HumanEval (Base)** | 68.3% | **86.0%** | 🏆 **7B** | **+17.7 pp** |
| **Pass@1 — HumanEval+ (Extra Tests)** | 62.8% | **82.9%** | 🏆 **7B** | **+20.1 pp** |
| **Inference Throughput (Tokens / Sec)** | **112.5 tok/s** | 24.0 tok/s | ⚡ **1.5B** | **4.7× faster** |
| **Time to First Token (TTFT)** | **106.6 ms** | 138.1 ms | ⚡ **1.5B** | **−31.5 ms** |
| **Peak GPU VRAM Usage** | **1.8 GB** | 4.4 GB | ⚡ **1.5B** | **2.4× less VRAM** |
| **Peak System RAM Usage** | 11.5 GB | **11.2 GB** | 🤝 Tie | ~250 MB |
| **Total Benchmark Wall-Clock Time** | **16 min 07 sec** | 27 min 29 sec | ⚡ **1.5B** | **1.7× faster** |

### Key Engineering Takeaway
* **Latency-Sensitive Workflows (IDE Autocomplete, Real-Time Assistants):** The **1.5B** model is the superior choice, delivering **112.5 tok/s** (4.7× faster) and occupying only **1.8 GB VRAM**.
* **Accuracy-Critical Batch Workflows (Automated Code Review, Unit Test Synthesis):** The **7B** model's **+20.1 percentage-point advantage on HumanEval+ (82.9%)** justifies its higher memory footprint while still fitting within a 6 GB laptop GPU.

---

## 🏗️ System Architecture

The benchmarking pipeline (`benchmark.py`) is structured around a decoupled, non-blocking 4-layer architecture:

1. **Orchestration Layer (`benchmark.py`):** Manages the sequential model lifecycle (load $\rightarrow$ warmup $\rightarrow$ HumanEval inference $\rightarrow$ regex sanitization $\rightarrow$ JSONL export $\rightarrow$ EvalPlus sandbox execution $\rightarrow$ unload).
2. **Inference Backend (LM Studio REST API v1):** Handles dynamic CUDA GPU offloading and exposes an OpenAI-compatible `/v1/chat/completions` streaming endpoint on `localhost:1234`.
3. **Output Sanitization Layer (`clean_completion`):** Uses targeted regular expressions to strip markdown code fences (` ```python ... ``` `) injected by instruction-tuned weights, preventing `SyntaxError` failures inside the EvalPlus sandboxed interpreter.
4. **System Monitoring Layer (`HardwareMonitor`):** Runs a background Python `daemon` thread polling `nvidia-smi` and `psutil` every 2.0 seconds so telemetry collection never skews TTFT or token throughput timings.

---

## 🛠️ Engineering Challenges Resolved

* **Resolving 0.0% Initial Pass@1 Rate:** Instruction-tuned models wrapped completions in Markdown backticks despite system-prompt instructions, causing `SyntaxError`s in EvalPlus. Implemented regex fence stripping (`clean_completion()`), restoring true evaluation accuracy (up to 86.0%).
* **Windows File-Descriptor Locking (`PermissionError`):** Bypassed OS-level stream locks on `NamedTemporaryFile` during multi-process EvalPlus execution by claiming unique temp file paths, closing the parent descriptor immediately, and writing sanitized JSONL streams in isolated UTF-8 blocks.
* **Sequential VRAM Protection:** Enforced strict single-model GPU residency with a 5-second memory settle delay between unloading the 1.5B model and loading the 7B model.

---

## 🚀 Getting Started

### 1. Prerequisites
* **LM Studio** installed and running the local server on port `1234`.
* Download the GGUF models inside LM Studio:
  * `qwen2.5-coder-1.5b-instruct` (`Q4_K_M`)
  * `qwen2.5-coder-7b-instruct` (`Q4_K_M`)
* NVIDIA GPU with `nvidia-smi` available in `PATH`.

### 2. Installation
```bash
git clone https://github.com/<your-username>/On-Device-LLM-Benchmarking.git
cd On-Device-LLM-Benchmarking
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the Benchmark
```bash
python benchmark.py
```

---

## 👥 Authors
* **Youssef Abdellatif** — *Department of Electronics and Computer Engineering, Nile University*
* **Rawda Mansour** — *Department of Electronics and Computer Engineering, Nile University*
