# 7. Modelos físicos y diagonalización exacta

## 7.1 El paquete `CouplingModel`

Cada modelo guarda términos de un sitio `h^o_i` y de pares `J^o_ij` y construye

```
H = Σ_o Σ_i h^o_i s^{a_o}_i  +  Σ_o Σ_{i<j} J^o_ij s^{a_o}_i s^{b_o}_j
```

con `s = σ` (Pauli, por defecto) o `s = S = σ/2` (`S_operators: true`: campos ×1/2,
acoplos ×1/4). Cada enlace cuenta una vez. Los coeficientes entran tal cual: J > 0 es
antiferromagnético. `ModelFactory(size, config1)` genera los puntos del barrido y
`cm.build_hamiltonian(hilbert, S_operators)` el operador de NetKet.

### Redes

| Red | NN | NN2 | NN3 |
|---|---|---|---|
| `Square(L0, L1, bc, order)` | (1,0), (0,1) | (1,1), (1,−1) | (2,0), (0,2) |
| `Triangular(L0, L1, a, b, bc, order)` | (1,0), (0,1), (−1,1) | (1,1), (−1,2), (2,−1) | (2,0), (0,2), (−2,2) |
| `Chain(size, bc, order)` | (1,0) | (2,0) | (3,0) |

`order` define la numeración de sitios: `default_1` (`u2·L0 + u1`), `default_2`
(`u1·L1 + u2`, filas; **es la que suponen las redes equivariantes y el ED por sectores**),
`snake_1`, `snake_2`. Con `bc: periodic` y L = 2 cada enlace aparece dos veces (2J),
comportamiento estándar de las condiciones periódicas.

### Modelos registrados

| Modelo | Red | Hamiltoniano (Pauli) | Parámetros |
|---|---|---|---|
| `J1J2Square` | Square | J1 Σ_NN σ·σ + J2 Σ_NN2 σ·σ + Σ_i (h_x σˣ + h_y σʸ + h_z σᶻ) | `J1`, `J2`, `fields=[hx,hy,hz]` |
| `SSMSquare` (Shastry–Sutherland) | Square | J Σ_NN σ·σ + Jp Σ_dímeros σ·σ + campos | `J` (enlace de la red cuadrada), `Jp` (dímero diagonal), `fields` (3 valores) |
| `IsingSquare` | Square | Σ_i h σ^{op}_i + J Σ_NN σᶻσᶻ | `J`, `fields=[h]`, `ops` (p. ej. `[["X"],["ZZ"]]`) |
| `LRSquare` / `LRChain` | Square / Chain | Σ_i (h_a σ^a + h_b σ^b) + Σ_{i<j} (J/𝒩) r_ij^{−α} σᶻσᶻ, con normalización de Kac 𝒩 = 1 + Σ_j r_{0j}^{−α} | `J`, `alpha`, `fields`, `ops` (fíjelo explícitamente, p. ej. `[["X","Z"],["ZZ"]]`); en LRChain `sizes: [[L,1]]` |
| `Oxalate` (J-K-Γ) | Triangular | J Σ σ·σ + términos de Kitaev y Γ según la dirección del enlace | `strength, theta, phi` (grados) o `J, K, Gamma` **con `"spherical": false`** |

Oxalate: con la parametrización esférica J = a sinθ cosφ, K = a sinθ sinφ, Γ = a cosθ. Por
dirección de enlace: (1,0) → K σˣσˣ, Γ(σʸσᶻ+σᶻσʸ); (0,1) → K σᶻσᶻ, Γ(σˣσʸ+σʸσˣ);
(−1,1) → K σʸσʸ, Γ(σˣσᶻ+σᶻσˣ).

SSM: los dímeros se cierran de forma periódica aunque `bc` sea `open`; aquí J es el enlace
de la red cuadrada y Jp el del dímero (al revés que en parte de la literatura).

### Barridos

Las claves `*_list` del bloque del modelo se barren (`mode: for` = producto cartesiano,
`zip` = emparejado). Cada combinación es un *punto*: comparte carpeta `UUID_…` con los
demás puntos del mismo tamaño y se distingue por el fragmento de parámetros del nombre de
fichero (p. ej. `_J1J2_1.0_0.5_XYZ_0.0_0.0_0.0`).

## 7.2 Diagonalización exacta (`Hydra/exact_diagonalization.py`)

`calc_exact_diag` se ejecuta al empezar cada punto si **N ≤ 20**:

* **Estado fundamental global** (`eigsh`, k = 1) dentro del espacio de Hilbert de la
  simulación (es decir, en el sector `sz_total` si está fijado).
* **Estado fundamental del sector** de la red, si el último stage fija un sector.

### Sector de cada stage

`stage_sectors` recorre los stages aplicando de forma acumulativa los `symm_wrapper` de cada
uno:

* traslaciones: `Traslation.irrep` si ese envoltorio está activo; si no, el `irrep` de un
  `Sequivariant`/`SequivariantX` que sea la **raíz** de la plantilla;
* Z2: se añade 0 (par, `trivial_Z2: true`) o 1 (impar).

Ejemplo: `(("Tx","Ty","Z2"), (1,1,0))`.

### ED por sectores sin matrices densas (`irrep_exact_diag`)

Las simetrías actúan como permutaciones de la base (traslación = `roll` sobre la malla
`(L0, L1)`; Z2 = inversión global, que exige `sz_total = 0`). Se usa el proyector
P = Π_g (1/|g|) Σ_n χ(q,n) U^n con χ_T = exp(+2πiqn/L) y χ_Z2 = (−1)^{qn}, y Lanczos sobre
A = H P + c(1−P) (c = cota de Gershgorin). La memoria es la de unos pocos vectores más H
dispersa. Un sector vacío produce un error.

### Comparación con la red

| Cantidad | Definición |
|---|---|
| `error_global` | \|E_best − E_gr_global\| / \|E_gr_global\| |
| `error_irrep` | \|E_best − E_gr_irrep\| / \|E_gr_irrep\| |
| `delta_irrep` | E_gr_irrep − E_gr_global |
| `keeper.bests[modo][k]` | Cada stage se compara con el ED de **su** sector |

`E_best` es la estimación Monte Carlo registrada en el mejor paso. Todas las energías van
en las unidades del Hamiltoniano. Referencias útiles (J2 = 0.5, Pauli): 4x4 −33.8317
(E/N = −0.5286 en unidades de S); 6x6 −72.5487 (E/N = −0.5038); 10x10 ≈ −199.1 (E/N ≈
−0.4977, literatura).

### Ficheros de ED

`<ED_label>.txt` (vector global, base de `hi`) y `<ED_label>irrep_[q…].txt` (vector del
sector, complejo; léase con `np.loadtxt(…, dtype=complex)`).

### Exploración del espectro (`ED_inspection.py`)

Recorre los tamaños y puntos de `config1_CM.json` en el espacio completo, calcula los 10
estados más bajos y guarda en `Figures/<modelo>/Size_LxL/<parámetros>/` el espectro,
módulo/fase de los 3 primeros estados, el histograma de magnetización y las contribuciones
acumuladas.
