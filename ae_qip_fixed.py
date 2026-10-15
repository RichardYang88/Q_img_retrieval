"""AE-QIP image-similarity estimation -- corrected, self-tested implementation.

Why the original notebook (AE-QIP.ipynb) reports a nonzero "Err rate" even in a
pure (noiseless) simulation -- three independent causes, in order of size:

 1. STATE-PREPARATION BUG in Us_gate (dominant).  Us_gate must load a
    normalized 8-dim image vector into 3 qubits with a uniformly-controlled
    RY tree (Mottonen-style):  RY(p0)@q2; CRY(p1) q2->q1 and CRY(p2) q2->q0...
    The original wiring applies p2 with the wrong control after the X-flip,
    applies p3 on the wrong (q2,q1) branch and uses a wrong X-flip sequence
    for p6, so the prepared state is NOT (|0>|e> + |1>|f>)/sqrt(2) and every
    downstream estimate is biased.  The corrected tree below is exact to
    machine precision (see --selftest).
 2. QAE GRID QUANTIZATION (~2.2% mean / 3.9% max at h=7 even with a perfect
    circuit): with h auxiliary qubits the phase estimate lives on the grid
    y/2^h, so the argmax read-out q_out(y*) = -cos(2*pi*y/2^h) carries an
    irreducible error ~ |ds/dy|/2 (h=8: ~1.3% mean, h=9: ~0.6%, h=10: ~0.3%).
 3. SHOT NOISE: shots=1 makes a single sample y the whole estimate.

How to get below 1% (validated analytically and on AerSimulator):
  * fix Us_gate (mandatory), and
  * either raise aux_qubit to h>=9 and keep the argmax read-out, or
  * keep h=7..8 and replace the argmax read-out by the maximum-likelihood
    read-out over the exact QAE likelihood
       P(y|s) = (1/2) sum_{sgn=+-} F_H(1/2 + sgn*theta(s)/pi - y/H),
       theta(s) = arcsin(sqrt((1-s)/2)), F_H(d) = sin^2(pi H d)/(H^2 sin^2(pi d))
    which interpolates between grid points using the whole histogram.
    Recommended: h=8, shots=1024 -> max Err ~0.17% (model), <=0.09% measured;
    h=7, shots=4096 works too.

Usage:
  python3 ae_qip_fixed.py --selftest
  python3 ae_qip_fixed.py --h 8 --shots 1024 [--mode shots|exact]
"""
import argparse
import numpy as np
from math import pi

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library import ZGate, RYGate
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

# data (same as the notebook / manuscript experiment)
x1 = np.array([0.325, 0.102, 0.28, 0.843, 0.16, 0.133, 0.125, 0.189])
x2 = np.array([[0.215, 0.004, 0.339, 0.857, 0.061, 0.17, 0.15, 0.223],
               [0.829, 0.153, 0.279, 0.161, 0.065, 0.041, 0.151, 0.397],
               [0.907, 0.16, 0.046, 0.052, 0.13, 0.202, 0.084, 0.289],
               [0.115, 0.032, 0.355, 0.068, 0.386, 0.391, 0.622, 0.408],
               [0.178, 0.345, 0.322, 0.114, 0.015, 0.004, 0.786, 0.339],
               [0.061, 0.081, 0.215, 0.028, 0.777, 0.481, 0.059, 0.322]])


def get_degree(x1, x2):
    """Angles of the uniformly-controlled RY trees for the two images."""
    e = x1 / np.linalg.norm(x1)
    f = x2 / np.linalg.norm(x2)
    a = lambda y: 2 * float(np.arccos(np.clip(y, -1, 1)))
    return [[a(np.linalg.norm(e[:4])), a(np.linalg.norm(e[4:6]) / np.linalg.norm(e[4:])),
             a(np.linalg.norm(e[:2]) / np.linalg.norm(e[:4])), a(e[2] / np.linalg.norm(e[2:4])),
             a(e[6] / np.linalg.norm(e[6:])), a(e[4] / np.linalg.norm(e[4:6])),
             a(e[0] / np.linalg.norm(e[:2]))],
            [a(np.linalg.norm(f[:4])), a(np.linalg.norm(f[4:6]) / np.linalg.norm(f[4:])),
             a(np.linalg.norm(f[:2]) / np.linalg.norm(f[:4])), a(f[2] / np.linalg.norm(f[2:4])),
             a(f[6] / np.linalg.norm(f[6:])), a(f[4] / np.linalg.norm(f[4:6])),
             a(f[0] / np.linalg.norm(f[:2]))]]


def Us_gate(params):
    """Load the params-encoded 8-dim unit vector into 3 qubits, EXACTLY.

    Uniformly-controlled RY tree: X-flips move the not-yet-addressed branches
    so that each controlled RY (ctrl_state='1..1') hits exactly its branch.
    Branch order: p0=root; p1=(q2=1); p2=(q2=0); p4=(1,1); p5=(1,0);
    p3=(0,1); p6=(0,0)  [(q2,q1) for the q0 level].
    """
    qc = QuantumCircuit(3, name="Us_gate")
    qc.ry(params[0], 2)                      # split e[0:4] | e[4:8]
    qc.cry(params[1], 2, 1)                  # branch q2=1
    qc.x(2)
    qc.cry(params[2], 2, 1)                  # branch q2=0
    qc.x(2)
    qc.append(RYGate(params[4]).control(2, ctrl_state='11'), [2, 1, 0])   # (1,1)
    qc.x(1)
    qc.append(RYGate(params[5]).control(2, ctrl_state='11'), [2, 1, 0])   # (1,0)
    qc.x(1)
    qc.x(2)
    qc.append(RYGate(params[3]).control(2, ctrl_state='11'), [2, 1, 0])   # (0,1)
    qc.x(2)
    qc.x([1, 2])
    qc.append(RYGate(params[6]).control(2, ctrl_state='11'), [2, 1, 0])   # (0,0)
    qc.x([1, 2])
    return qc


def Us_layer(params):
    """Embedding U_phi: |0000> -> (|0>(|e>+|f>) + |1>(|e>-|f>))/2."""
    qc = QuantumCircuit(4, name="Us_layer")
    qc.h(0)
    qc.compose(Us_gate(params[0]).control(1, None, '1'), [0, 1, 2, 3], inplace=True)
    qc.x(0)
    qc.compose(Us_gate(params[1]).control(1, None, '1'), [0, 1, 2, 3], inplace=True)
    qc.x(0)
    qc.h(0)
    return qc


def Ug_layer(params):
    """Grover operator Q = U_phi (I - 2|0><0|) U_phi^dagger . Z_flag."""
    qc = QuantumCircuit(4, name="Ug_gate")
    qc.z(0)
    qc.compose(Us_layer(params).inverse(), range(4), inplace=True)
    qc.x(range(4))
    cccz = ZGate().control(num_ctrl_qubits=3, ctrl_state='111')
    qc.compose(cccz, [1, 2, 3, 0], inplace=True)
    qc.x(range(4))
    qc.compose(Us_layer(params), range(4), inplace=True)
    return qc

# --------------------------------------------------------------------------
# QAE circuit (identical to the notebook)
# --------------------------------------------------------------------------
def swap_registers(circuit, n):
    for qubit in range(n // 2):
        circuit.swap(qubit, n - qubit - 1)
    return circuit


def qft_rotations(circuit, n):
    if n == 0:
        return circuit
    n -= 1
    circuit.h(n)
    for qubit in range(n):
        circuit.cp(pi / 2 ** (n - qubit), qubit, n)
    qft_rotations(circuit, n)


def qft(circuit, n):
    qft_rotations(circuit, n)
    swap_registers(circuit, n)
    return circuit


def inverse_qft(num_qubits):
    c = qft(QuantumCircuit(num_qubits, name="QFT"), num_qubits).inverse()
    qc = QuantumCircuit(num_qubits)
    qc.append(c, range(num_qubits))
    return qc


def GQAE_layer(orcal_qubit, aux_qubit, params):
    num_qubits = orcal_qubit + aux_qubit
    qr = QuantumRegister(num_qubits, 'q')
    cr = ClassicalRegister(aux_qubit, 'c')
    qc = QuantumCircuit(qr, cr)
    qc.h(range(aux_qubit))
    qc.append(Us_layer(params), range(aux_qubit, num_qubits))
    reps = 1
    for i in range(aux_qubit):
        for _ in range(reps):
            qc.append(Ug_layer(params).control(1, None, '1'),
                      [i] + np.arange(aux_qubit, num_qubits).tolist())
        reps *= 2
    qc.append(inverse_qft(aux_qubit), range(aux_qubit))
    qc.measure(qr[:aux_qubit], cr[:aux_qubit])
    return qc


# --------------------------------------------------------------------------
# read-outs
# --------------------------------------------------------------------------
def qae_dist(s_arr, h):
    """Exact QAE output distribution P(y|s), y=0..2^h-1 (Fejer kernel mixture)."""
    Hh = 2 ** h
    y = np.arange(Hh)
    s_arr = np.atleast_1d(np.asarray(s_arr, float))
    th = np.arcsin(np.sqrt(np.clip((1.0 - s_arr) / 2.0, 0, 1)))
    out = np.zeros((len(th), Hh))
    for sgn in (+1, -1):
        phi = (0.5 + sgn * th / pi)[:, None]
        d = phi - y[None, :] / Hh
        den = np.sin(pi * d) ** 2
        num = np.sin(pi * (Hh * phi - y[None, :])) ** 2
        out += 0.5 * np.where(den > 1e-14, num / np.maximum(den, 1e-300), float(Hh)) / Hh ** 2
    return out


_COARSE = np.linspace(-0.9999, 0.9999, 2001)


def mle_readout(counts_vec, h):
    """Maximum-likelihood similarity estimate from a QAE histogram."""
    c = np.asarray(counts_vec, float)
    c = c / c.sum()
    Lc = np.log(np.maximum(qae_dist(_COARSE, h), 1e-300))
    s0 = _COARSE[int(np.argmax(Lc @ c))]
    fine = np.linspace(max(-1.0, s0 - 0.004), min(1.0, s0 + 0.004), 401)
    Lf = np.log(np.maximum(qae_dist(fine, h), 1e-300))
    return float(fine[int(np.argmax(Lf @ c))])


def counts_to_vec(counts, h):
    vec = np.zeros(2 ** h)
    for k, v in counts.items():
        vec[int(k.replace(' ', ''), 2)] += v
    return vec

# --------------------------------------------------------------------------
# self-test: the corrected Us_gate must prepare the image vector exactly
# --------------------------------------------------------------------------
def self_test():
    ok = True
    for i in range(len(x2)):
        params = get_degree(x1, x2[i])
        for vec, tag, idx in ((x1, 'e', 0), (x2[i], 'f', 1)):
            target = vec / np.linalg.norm(vec)
            sv = Statevector(Us_gate(params[idx])).data
            dev = float(np.max(np.abs(sv - target)))
            ok &= dev < 1e-10
            print(f"SN{i+1} Us_gate[{tag}] max |amplitude - target| = {dev:.3e}"
                  f"  {'OK' if dev < 1e-10 else 'FAIL'}")
    print("SELF-TEST", "PASSED" if ok else "FAILED")
    return ok


# --------------------------------------------------------------------------
# experiment driver
# --------------------------------------------------------------------------
def run(h, shots, mode):
    theta = 2 ** (h - 1)
    q_out = lambda y: round(float(-np.cos(pi * y / theta)), 4)
    backend = AerSimulator(method="statevector")
    print(f"AE-QIP fixed | h={h} ({h+4} qubits) | shots={shots} | mode={mode}")
    hdr = f"{'SN':<4}{'y*':<6}{'argmax q_out':<14}{'Err%':<10}{'MLE s':<12}{'MLE Err%':<10}{'Classic':<9}"
    print(hdr)
    print("-" * len(hdr))
    ea, em = [], []
    for i in range(len(x2)):
        params = get_degree(x1, x2[i])
        qc = transpile(GQAE_layer(4, h, params), backend, optimization_level=0)
        if mode == "exact":
            sv = Statevector.from_instruction(
                qc.remove_final_measurements(inplace=False))
            p = sv.probabilities(qargs=list(range(h)))
            vec = p * 1024.0        # MLE only needs the histogram shape
        else:
            vec = counts_to_vec(backend.run(qc, shots=shots).result().get_counts(), h)
        y = int(np.argmax(vec))
        qo = q_out(y)
        co = round(float(np.matmul(x1, x2[i])), 4)
        sm = round(mle_readout(vec, h), 6)
        ea.append(100 * abs(qo - co) / abs(co))
        em.append(100 * abs(sm - co) / abs(co))
        print(f"{i+1:<4}{y:<6}{qo:<14}{ea[-1]:<10.4f}{sm:<12}{em[-1]:<10.4f}{co:<9}")
    print("-" * len(hdr))
    print(f"argmax read-out : mean {np.mean(ea):.4f}%  max {np.max(ea):.4f}%")
    print(f"MLE read-out    : mean {np.mean(em):.4f}%  max {np.max(em):.4f}%")
    return float(np.max(em))


def main():
    ap = argparse.ArgumentParser(description="Corrected AE-QIP pipeline")
    ap.add_argument("--h", type=int, default=8, help="auxiliary qubits (default 8)")
    ap.add_argument("--shots", type=int, default=1024)
    ap.add_argument("--mode", choices=["shots", "exact"], default="shots")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        raise SystemExit(0 if self_test() else 1)
    raise SystemExit(0 if run(args.h, args.shots, args.mode) < 1.0 else 1)


if __name__ == "__main__":
    main()
