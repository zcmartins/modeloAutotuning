# autotuner_exec_gui_pso_cma_hybrid.py

"""
Autotuner GUI - PSO / CMA-ES / Híbrido PSO→CMA-ES
Parada segura com stop_event.
"""

import os
import subprocess
import threading
import queue
import time
import random
import statistics
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog
from concurrent.futures import ThreadPoolExecutor, as_completed

# import CMA-ES
try:
    import cma
    CMA_AVAILABLE = True
except ImportError:
    CMA_AVAILABLE = False

# -------------------------
# Utilities
# -------------------------
def parse_output_for_number(output: str):
    if output is None:
        raise ValueError("No output")
    out = output.replace(",", ".")
    for line in out.splitlines():
        low = line.lower()
        if 'score' in low or 'value' in low or 'resultado' in low:
            import re
            m = re.search(r'([-+]?\d*\.\d+|\d+)', line)
            if m:
                return float(m.group(0))
    import re
    m = re.search(r'([-+]?\d*\.\d+|\d+)', out)
    if m:
        return float(m.group(0))
    raise ValueError("Nenhum número encontrado na saída do executável.")

def run_executable_capture(exe_path: str, params: list, timeout: float = 30.0):
    try:
        exe_path = os.path.abspath(exe_path)
        if not os.path.isfile(exe_path):
            return None
        cmd = [exe_path] + [str(p) for p in params]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True, timeout=timeout, shell=False)
        out = (proc.stdout or "") + "\n" + (proc.stderr or "")
        try:
            return parse_output_for_number(out)
        except Exception:
            return None
    except subprocess.TimeoutExpired:
        return None
    except Exception:
        return None

def make_objective(exe_path, param_defs, maximize=True, timeout=30.0):
    def decode_and_eval(vec):
        params = []
        for v, spec in zip(vec, param_defs):
            low, high = spec['low'], spec['high']
            val = max(low, min(high, v))
            if spec['type'] == 'int':
                val = int(round(val))
            params.append(val)
        score = run_executable_capture(exe_path, params, timeout=timeout)
        if score is None:
            return float('-inf') if maximize else float('inf')
        return score if maximize else -score
    return decode_and_eval

def should_stop_early(history, population=None, min_delta=1e-6, patience=25, std_threshold=1e-3):
    if len(history) >= patience + 1:
        recent = history[-(patience+1):]
        improvement = max(recent) - min(recent)
        if improvement < min_delta:
            return True
    if population is not None and len(population) > 1:
        dim = len(population[0])
        converged = True
        for d in range(dim):
            vals = [ind[d] for ind in population]
            stdev = statistics.pstdev(vals)
            rng = max(1e-12, max(vals) - min(vals))
            if stdev > std_threshold * rng:
                converged = False
                break
        if converged:
            return True
    return False

# -------------------------
# PSO
# -------------------------
def pso_optimize(obj_fn, dim, bounds, executor, stop_event,
                 n_particles=None, max_iter=200, patience=25):
    import numpy as np
    if n_particles is None:
        n_particles = max(10, 4*dim)

    w_max, w_min = 0.9, 0.4
    c1, c2 = 2.0, 2.0

    particles = [np.array([np.random.uniform(b[0], b[1]) for b in bounds]) for _ in range(n_particles)]
    velocities = [np.zeros(dim) for _ in range(n_particles)]
    pbest = particles.copy()
    pbest_scores = [float('-inf')]*n_particles

    # Avalia população inicial
    for i, p in enumerate(particles):
        pbest_scores[i] = obj_fn(p)

    gbest_idx = int(np.argmax(pbest_scores))
    gbest = pbest[gbest_idx].copy()
    gbest_score = pbest_scores[gbest_idx]
    history = [gbest_score]

    for it in range(max_iter):
        if stop_event.is_set():
            break
        w = w_max - (w_max - w_min) * it / max_iter  # linear decay
        for i in range(n_particles):
            r1, r2 = np.random.rand(dim), np.random.rand(dim)
            velocities[i] = w*velocities[i] + c1*r1*(pbest[i]-particles[i]) + c2*r2*(gbest-particles[i])
            # limitar velocidade
            for d in range(dim):
                v_max = (bounds[d][1]-bounds[d][0])*0.2
                velocities[i][d] = np.clip(velocities[i][d], -v_max, v_max)
            particles[i] += velocities[i]
            # respeitar bounds
            for d in range(dim):
                particles[i][d] = np.clip(particles[i][d], bounds[d][0], bounds[d][1])
            score = obj_fn(particles[i])
            if score > pbest_scores[i]:
                pbest[i] = particles[i].copy()
                pbest_scores[i] = score
            if score > gbest_score:
                gbest = particles[i].copy()
                gbest_score = score
        history.append(gbest_score)
        if len(history)>patience and max(history[-patience:])-min(history[-patience:])<1e-6:
            break
    return gbest, gbest_score, history

# -------------------------
# CMA-ES
# -------------------------
def cmaes_optimize(obj_fn, dim, bounds, stop_event, max_iter=300):
    import numpy as np
    if not CMA_AVAILABLE:
        raise RuntimeError("CMA-ES não disponível")
    x0 = np.array([(b[0]+b[1])/2 for b in bounds])
    sigma0 = max([(b[1]-b[0])*0.3 for b in bounds])
    opts = {'bounds': [[b[0] for b in bounds],[b[1] for b in bounds]],
            'popsize': 4 + int(3*np.log(dim))}
    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
    history = []
    while not es.stop() and not stop_event.is_set():
        solutions = es.ask()
        scores = [obj_fn(s) for s in solutions]
        es.tell(solutions, [-s for s in scores])
        history.append(max(scores))
    best_idx = int(np.argmax(scores))
    return solutions[best_idx], scores[best_idx], history


# -------------------------
# Híbrido PSO->CMA
# -------------------------
def hybrid_optimize(obj_fn, dim, bounds, executor, stop_event,
                    pso_particles=None, pso_iters=50, cma_maxiter=200, patience=25):
    pso_best, pso_score, pso_hist = pso_optimize(obj_fn, dim, bounds, executor, stop_event,
                                                n_particles=pso_particles, max_iter=pso_iters, patience=patience)
    if stop_event.is_set():
        return pso_best, pso_score, {'pso_hist': pso_hist, 'cma_hist': []}
    cma_best, cma_score, cma_hist = cmaes_optimize(obj_fn, dim, bounds, stop_event, max_iter=cma_maxiter)
    if cma_score >= pso_score:
        return cma_best, cma_score, {'pso_hist': pso_hist, 'cma_hist': cma_hist}
    else:
        return pso_best, pso_score, {'pso_hist': pso_hist, 'cma_hist': cma_hist}


# -------------------------
# GUI Application
# -------------------------
class AutoTunerApp:
    def __init__(self, root):
        self.root = root
        root.title("Autotuner - PSO / CMA-ES / Híbrido")
        self.frame = ttk.Frame(root, padding=8)
        self.frame.pack(fill=tk.BOTH, expand=True)

        # Executable
        ttk.Label(self.frame, text="Caminho do executável (.exe):").grid(row=0, column=0, sticky=tk.W)
        self.exe_entry = ttk.Entry(self.frame, width=60)
        self.exe_entry.grid(row=0, column=1, columnspan=3, sticky=tk.W)
        ttk.Button(self.frame, text="Procurar", command=self.browse_exe).grid(row=0, column=4)

        # n params
        ttk.Label(self.frame, text="Quantidade de parâmetros:").grid(row=1, column=0, sticky=tk.W)
        self.nparams_var = tk.IntVar(value=1)
        ttk.Entry(self.frame, textvariable=self.nparams_var, width=6).grid(row=1, column=1, sticky=tk.W)
        ttk.Button(self.frame, text="Gerar campos", command=self.generate_param_fields).grid(row=1, column=2, sticky=tk.W)

        # params frame
        self.params_frame = ttk.Frame(self.frame)
        self.params_frame.grid(row=2, column=0, columnspan=5, pady=6, sticky=tk.W)

        # method selection
        ttk.Label(self.frame, text="Método:").grid(row=3, column=0, sticky=tk.W)
        self.method_var = tk.StringVar(value="pso")
        ttk.Radiobutton(self.frame, text="PSO", variable=self.method_var, value="pso").grid(row=3, column=1, sticky=tk.W)
        ttk.Radiobutton(self.frame, text="CMA-ES", variable=self.method_var, value="cma").grid(row=3, column=2, sticky=tk.W)
        ttk.Radiobutton(self.frame, text="Híbrido PSO→CMA", variable=self.method_var, value="hybrid").grid(row=3, column=3, sticky=tk.W)

        # objective
        ttk.Label(self.frame, text="Objetivo:").grid(row=4, column=0, sticky=tk.W)
        self.max_var = tk.BooleanVar(value=True)
        ttk.Radiobutton(self.frame, text="Maximizar", variable=self.max_var, value=True).grid(row=4, column=1, sticky=tk.W)
        ttk.Radiobutton(self.frame, text="Minimizar", variable=self.max_var, value=False).grid(row=4, column=2, sticky=tk.W)

        # workers & patience
        ttk.Label(self.frame, text="Workers:").grid(row=5, column=0, sticky=tk.W)
        self.workers_var = tk.IntVar(value=4)
        ttk.Entry(self.frame, textvariable=self.workers_var, width=6).grid(row=5, column=1, sticky=tk.W)
        ttk.Label(self.frame, text="Patience (it):").grid(row=5, column=2, sticky=tk.W)
        self.patience_var = tk.IntVar(value=25)
        ttk.Entry(self.frame, textvariable=self.patience_var, width=6).grid(row=5, column=3, sticky=tk.W)

        # controls
        ttk.Button(self.frame, text="Tudo = int", command=self.set_all_int).grid(row=6, column=0, pady=6)
        ttk.Button(self.frame, text="Tudo = float", command=self.set_all_float).grid(row=6, column=1, pady=6)
        ttk.Button(self.frame, text="Setar Min/Max (todos)", command=self.set_all_bounds).grid(row=6, column=2, pady=6)

        ttk.Button(self.frame, text="Iniciar Otimização", command=self.start_optimization_thread).grid(row=7, column=0, pady=8)
        ttk.Button(self.frame, text="Parar", command=self.stop_optimization).grid(row=7, column=1, pady=8)

        # output log
        self.output = tk.Text(self.frame, height=20, width=100)
        self.output.grid(row=8, column=0, columnspan=5, pady=6)

        self.param_widgets = []
        self.log_queue = queue.Queue()
        self.root.after(100, self._process_log_queue)

        # control members
        self.optim_thread = None
        self.stop_event = threading.Event()
        self.executor = None

    # logging
    def _process_log_queue(self):
        try:
            while True:
                item = self.log_queue.get_nowait()
                self.output.insert(tk.END, item + "\n")
                self.output.see(tk.END)
        except queue.Empty:
            pass
        self.root.after(100, self._process_log_queue)

    def log(self, *args):
        self.log_queue.put(" ".join(map(str, args)))

    # browse
    def browse_exe(self):
        path = filedialog.askopenfilename(title="Selecione o executável")
        if path:
            self.exe_entry.delete(0, tk.END)
            self.exe_entry.insert(0, path)

    # param fields
    def generate_param_fields(self):
        for w in self.param_widgets:
            for widget in w:
                widget.destroy()
        self.param_widgets.clear()
        n = max(1, self.nparams_var.get())
        ttk.Label(self.params_frame, text="Idx").grid(row=0, column=0)
        ttk.Label(self.params_frame, text="Tipo").grid(row=0, column=1)
        ttk.Label(self.params_frame, text="Min").grid(row=0, column=2)
        ttk.Label(self.params_frame, text="Max").grid(row=0, column=3)
        for i in range(n):
            idx_lbl = ttk.Label(self.params_frame, text=str(i+1))
            type_cb = ttk.Combobox(self.params_frame, values=["float", "int"], width=8)
            type_cb.set("float")
            min_entry = ttk.Entry(self.params_frame, width=12)
            max_entry = ttk.Entry(self.params_frame, width=12)
            min_entry.insert(0, "-10")
            max_entry.insert(0, "10")
            idx_lbl.grid(row=i+1, column=0)
            type_cb.grid(row=i+1, column=1)
            min_entry.grid(row=i+1, column=2)
            max_entry.grid(row=i+1, column=3)
            self.param_widgets.append((idx_lbl, type_cb, min_entry, max_entry))

    def set_all_int(self):
        if not self.param_widgets:
            messagebox.showinfo("Info", "Gere os campos primeiro (Gerar campos).")
            return
        for _, type_cb, _, _ in self.param_widgets:
            type_cb.set("int")
        self.log("Todos parâmetros definidos como int.")

    def set_all_float(self):
        if not self.param_widgets:
            messagebox.showinfo("Info", "Gere os campos primeiro (Gerar campos).")
            return
        for _, type_cb, _, _ in self.param_widgets:
            type_cb.set("float")
        self.log("Todos parâmetros definidos como float.")

    def set_all_bounds(self):
        if not self.param_widgets:
            messagebox.showinfo("Info", "Gere os campos primeiro (Gerar campos).")
            return
        s = simpledialog.askstring("Setar Min/Max", "Digite dois valores separados por vírgula (ex: -5,5) ou clique Cancel para entrada separada:")
        if s:
            try:
                a,b = s.split(",")
                low = float(a.strip()); high = float(b.strip())
            except Exception:
                messagebox.showerror("Erro", "Formato inválido. Use: -5,5")
                return
        else:
            low = simpledialog.askfloat("Min", "Valor Min para todos:")
            if low is None: return
            high = simpledialog.askfloat("Max", "Valor Max para todos:")
            if high is None: return
        if low > high:
            messagebox.showerror("Erro", "Min maior que Max.")
            return
        for _,_,min_entry,max_entry in self.param_widgets:
            min_entry.delete(0, tk.END); min_entry.insert(0, str(low))
            max_entry.delete(0, tk.END); max_entry.insert(0, str(high))
        self.log(f"Todos bounds definidos: {low} .. {high}")

    # thread control
    def start_optimization_thread(self):
        if self.optim_thread and self.optim_thread.is_alive():
            messagebox.showinfo("Info", "Otimização já em execução.")
            return
        self.stop_event.clear()
        self.optim_thread = threading.Thread(target=self.start_optimization, daemon=True)
        self.optim_thread.start()

    def stop_optimization(self):
        if self.optim_thread and self.optim_thread.is_alive():
            self.log("Pedido de parada recebido. Parando...")
            self.stop_event.set()
        else:
            self.log("Nenhuma otimização em execução.")

    # main optimize
    def start_optimization(self):
        exe = self.exe_entry.get().strip()
        if not exe:
            self.log("Erro: informe o caminho do executável.")
            messagebox.showerror("Erro", "Informe o caminho do executável.")
            return

        param_defs = []
        for widgets in self.param_widgets:
            _, type_cb, min_entry, max_entry = widgets
            t = type_cb.get().strip()
            try:
                low = float(min_entry.get().strip())
                high = float(max_entry.get().strip())
            except Exception:
                self.log("Bounds inválidos.")
                messagebox.showerror("Erro", "Bounds inválidos.")
                return
            if low > high:
                self.log("Min > Max em algum parâmetro.")
                messagebox.showerror("Erro", "Min maior que Max.")
                return
            param_defs.append({'type': t, 'low': low, 'high': high})

        if not param_defs:
            self.log("Crie os parâmetros primeiro.")
            messagebox.showerror("Erro", "Crie os parâmetros primeiro (Gerar campos).")
            return

        method = self.method_var.get()
        maximize = self.max_var.get()
        workers = max(1, int(self.workers_var.get()))
        patience = max(1, int(self.patience_var.get()))
        timeout_eval = 20.0  # seconds per executable evaluation

        obj = make_objective(exe, param_defs, maximize=maximize, timeout=timeout_eval)
        bounds = [(d['low'], d['high']) for d in param_defs]
        dim = len(bounds)

        # create executor
        self.executor = ThreadPoolExecutor(max_workers=workers)

        self.log("Iniciando otimização...")
        t0 = time.time()

        best_vec = None
        best_score_internal = float('-inf')

        try:
            if method == 'pso':
                self.log("Executando PSO ...")
                best_vec, best_score_internal, hist = pso_optimize(
                    obj, dim, bounds, executor=self.executor, stop_event=self.stop_event,
                    n_particles=max(8, 2*dim+4), max_iter=200, patience=patience)

            elif method == 'cma':
                self.log("Executando CMA-ES ...")
                best_vec, best_score_internal, hist = cmaes_optimize(
                    obj, dim, bounds, stop_event=self.stop_event, max_iter=300)

            else:
                self.log("Executando Híbrido PSO→CMA-ES ...")
                best_vec, best_score_internal, meta = hybrid_optimize(
                    obj, dim, bounds, executor=self.executor, stop_event=self.stop_event,
                    pso_particles=max(6, 2*dim+2), pso_iters=40, cma_maxiter=200, patience=patience)

            if best_vec is None:
                best_vec = [(b[0]+b[1])/2.0 for b in bounds]
                best_score_internal = float('-inf')

            best_params = []
            for v, spec in zip(best_vec, param_defs):
                val = int(round(v)) if spec['type'] == 'int' else v
                best_params.append(val)

            elapsed = time.time() - t0
            if self.stop_event.is_set():
                self.log("Otimização interrompida pelo usuário. Resultado parcial retornado.")
            self.log("=== RESULTADO ===")
            self.log("Melhor(es) parâmetros encontrados:", best_params)
            self.log("Score (orig):", best_score_internal if maximize else -best_score_internal)
            self.log(f"Tempo total: {elapsed:.2f} s")
            self.log("=================")

        except Exception as e:
            self.log("Erro durante otimização:", e)
            messagebox.showerror("Erro", f"Erro durante otimização: {e}")
        finally:
            try:
                if self.executor:
                    self.executor.shutdown(wait=False)
            except Exception:
                pass
            self.executor = None
            self.stop_event.clear()

# run
if __name__ == "__main__":
    root = tk.Tk()
    app = AutoTunerApp(root)
    app.generate_param_fields()
    root.mainloop()
