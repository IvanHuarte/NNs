# 4. Redes neuronales: el constructor Hydra

El paquete `Hydra/NN/` convierte una descripción en JSON (config2 + config3) en un modelo
flax listo para NetKet, y permite que la arquitectura **evolucione por etapas** (*stages*)
durante una misma simulación, trasplantando los pesos compatibles de una etapa a la
siguiente.

```
config2_Hydra.json  ──►  plantilla (template) de cada stage ┐
config3_Hydra_NN.json ─►  storage (hiperparámetros) +        ├─► Hydra ──► modelo flax
                          symm_wrapper (Z2 / Traslation)     ┘           (NNWrapper)
```

## 4.1 Conceptos

| Concepto | Qué es | Dónde se define |
|---|---|---|
| **Plantilla** (*template*) | Árbol de nombres de módulos que fija la topología de la red | `config2[selection]["stage<k>"]["template"]` |
| **Storage** | Hiperparámetros de cada clase de módulo (campos del dataclass flax) | `config3["storage"]` |
| **Módulo de flujo** (*flux*) | Combina submódulos: secuencia, proyección de momento, ramas paralelas… | `Hydra/NN/NN/FluxModules/` |
| **Módulo simple** (*single*) | Bloque de red: backbone (ViT2D, CvT3, CNN…) o cabeza (OutputHead…) | `Hydra/NN/NN/SingleModules/` |
| **Envoltorio de simetría** | Proyecta sobre un sector de Z2 o de traslaciones | `config3["symm_wrapper"]`, `Hydra/NN/NN/Symm/` |
| **Stage / eon** | Una arquitectura (y un sector) entrenada durante un bloque del schedule | `stage0`, `stage1`, … |
| **Código** (*code*) | Cadena de dígitos que identifica un nodo del árbol de parámetros (`"0"`, `"01"`, `"A"`) | Generado por Hydra; se usa en `load` y en el schedule |

### Esquema de config2

```json
{
  "selection": "CvT3OH000",
  "load_model": {"parameters": false, "sampler": false, "checkpoint": 0, "load_from": []},
  "CvT3OH000": {
    "stage0": {"template": {...}, "save_params": true},
    "stage1": {"template": {...}, "load": [["stage0", "A", "A"]],
               "change_attr": {...}, "symm_wrapper": {"Z2": {"trivial_Z2": false}}}
  }
}
```

Sólo cuenta la entrada indicada por `selection`; el resto del fichero es almacén personal
de experimentos. Convención de nombres: `<Módulo><Cabeza><n_y n_x z2>` más un sufijo
para variantes, p. ej. `CvT3OH221` = CvT3 + OutputHead en el sector k = (2,2), Z2 impar
(ver `Hydra.label_utils.experiment_name`).

### Esquema de config3

```json
{
  "print_arch": false,
  "symm_wrapper": {
    "Traslation": {"on": false, "irrep": [0, 1], "use_anchor": false, "save_memory": false},
    "Z2": {"on": true, "trivial_Z2": true, "symm_modulus": false}
  },
  "storage": {
    "CvT3": {"n_CP_blocks": [1, 3], "channels": [30, 60], "attn_heads": [2, 4],
             "kernel": [[3, 3], [3, 3]], "strides": [[2, 2], [1, 1]], "final_architecture": null},
    "OutputHead": {"d_model": 60, "only_phase": false}
  }
}
```

config2 y config3 se fusionan como `{**config2, **config3}` (`VMC_simulation.py`); si una
clave coincide, gana config3.

## 4.2 Plantillas

Una plantilla tiene **una sola clave raíz**. Reglas (`Hydra/NN/utils.py`):

* **Clave de flujo** (`Sequential`, `Sequivariant`, `SequivariantX`, `Transversal`,
  `SplitTraining`, `SingleModule`): su valor es un dict. Las entradas cuyo valor es un dict
  son submódulos (en orden de inserción); las que no lo son, hiperparámetros del flujo
  (`"irrep"`, `"coset"`, `"phase_mode"`, `"operation"`…). Los hijos se renombran por
  posición: `Seq_0, Seq_1…` (Sequential), `SeqV_i` (Sequivariant), `Trans_i`
  (Transversal), `Modulus`/`Phase` (SplitTraining), `Single` (SingleModule).
* **Clave de módulo simple** con valor `{}`: los hiperparámetros se copian de
  `storage[<nombre>]`.
  * Sufijos: `"MLP_2"` u `"OutputHead_b"` permiten repetir un módulo bajo el mismo objeto
    JSON; la clase **y** la entrada de storage se toman del nombre base (`storage["MLP"]`).
    Una entrada de storage con sufijo nunca se lee.
  * **No se admiten hiperparámetros en línea** dentro de la plantilla
    (`{"CvT3": {"channels": [4]}}` crea un módulo falso llamado `channels`). Para que dos
    copias de un módulo difieran, use `change_attr` (§4.5).
* **Referencia a otra etapa**: `{"stage0": {}}` reutiliza la red desnuda de esa etapa (sin
  envoltorios de simetría, que se vuelven a añadir).

Ejemplo (la red estándar de las campañas actuales):

```json
"template": {"Sequivariant": {"CvT3": {}, "OutputHead": {}, "irrep": [0, 0],
                              "phase_mode": "angle", "coset": [2, 2]}}
```

### Restricciones de nombres

Los hijos de un nodo se detectan porque su clave contiene (sensible a mayúsculas) alguna de
las subcadenas `Wrap, ARNN, NN, Single, Modulus, Phase, Seq, SeqV, SeqVX, Trans`. Por tanto
**ningún hiperparámetro puede contener esas subcadenas** (`phase_mode` u `only_phase` valen
porque van en minúscula). Además, cualquier clave que contenga `activation` y cuyo valor
sea una lista se convierte en una tupla de funciones (`utils.activation_dict`); por eso el
campo de activación de CvT3 se llama `ffn_act`.

## 4.3 De la plantilla al modelo

1. **Envoltorio principal**: `NNWrapper` (o `ARNNWrapper` si hay un módulo autorregresivo).
   `NNWrapper` recibe `(B, N)` y lo reordena a `(B, L0, L1, 1)`; devuelve `(B,)`.
   **Convención de ejes**: el eje 1 es la primera dimensión de la red (filas, «y») y el eje
   2 la segunda («x»); los irreps se escriben `(n_y, n_x)`.
2. **Envoltorios de simetría**: por cada entrada de `symm_wrapper` con `"on": true`, en el
   orden del dict (el primero es el más externo). Con el orden de config3: `Traslation(Z2(red))`.
3. **Red interior**: cada nodo `{"module", "setup"}` se instancia con su clase del
   registro. Se inyectan automáticamente `lattice_size` y/o `hilbert` a los módulos que los
   necesitan.

### Códigos de submódulo (code2path)

Cada nivel del árbol de parámetros añade un dígito (posición del hijo). Los niveles `NN`,
`ARNN`, `Z2Wrap` y `TWrap` no añaden dígito, de modo que activar Z2 no cambia los códigos.

| Red | Código | Ruta |
|---|---|---|
| Sequential(ViT2D, Head) | `"0"` | `("NN","Seq_0")` |
| Sequential(ViT2D, Head) | `"1"` | `("NN","Seq_1")` |
| Igual con Z2 | `"0"` | `("NN","Z2Wrap","Seq_0")` |
| — | `"A"` | Toda la red |

Hydra imprime al construir cada etapa un diagrama de cajas con el código de cada módulo
(`arch_diagram.py`), el árbol de parámetros (`print_arch: true` muestra todos los nodos) y
el número de parámetros. Un código inválido produce un error que lista los válidos.

## 4.4 Módulos de flujo

| Módulo | Campos (defecto) | Operación |
|---|---|---|
| `Sequential` | — | `x = m(x)` en orden |
| `SingleModule` | `mode=None` | `None`: identidad; `"modulus"`: `x + 0i`; `"phase"`: `1 + i·x` |
| `SplitTraining` | — | `Modulus(x) + 1j·Phase(x)` (ambos `(B,1)`) |
| `Transversal` | `operation="sum"`, `post_norm=False` | Ramas paralelas combinadas: `sum`, `sum_angles`, `product`, `mean`, `modulus_marshall`, `None` |
| `Sequivariant` | `irrep` (oblig.), `phase_mode="angle"`, `coset=None`, `eps`, `z_eps` | Proyección de momento (abajo) |
| `SequivariantX` | `irrep`, `eps` | Como Sequivariant, pero la fase sale de la configuración de entrada (no variacional) |

### Sequivariant: proyección sobre un momento

1. Todos los submódulos salvo el último forman el *backbone*, que debe devolver una malla
   `(B, H, W, C)` equivariante bajo traslaciones.
2. Con los caracteres χ_k(r) = exp(2πi(n_y r_y/H + n_x r_x/W)) sobre la malla de salida se
   forma z = Σ_r χ_k(r)·x[..., 0] (sólo el canal 0).
3. El último submódulo (la cabeza) da la log-amplitud.
4. `phase_mode`:
   * `"angle"` (por defecto): log ψ = cabeza + i·arg(z). La amplitud no se anula donde
     z → 0.
   * `"log"`: log ψ = cabeza + log z, es decir ψ = e^cabeza·z; ψ se anula de forma continua
     en z = 0. Mismos parámetros que `angle` (checkpoints intercambiables).

**`coset = (s_y, s_x)`**: la entrada se desplaza por todos los t ∈ {0..s_y−1}×{0..s_x−1} y
se combina ψ_k(x) = Σ_t e^{−ik·t} ψ(T_t x). Coste: s_y·s_x pasadas.

| | Sin coset | Con coset = stride total |
|---|---|---|
| Significado de `irrep` | Momento en la malla de salida (k = 2πn/H) | Momento de la red completa (k = 2πn/L) |
| Simetría exacta | Sólo traslaciones por el stride (T_s); se mezclan k y k + 2πa/s | Autoestado exacto de momento de toda la red |

Condiciones de exactitud: backbone equivariante con padding circular (verificado por tests
para CvT3, CvT, ViT2D y CNNLiang con `window_pooling=1`); cabeza invariante (OutputHead,
DeepOutputHead y ComplexHead suman sobre tokens); lados divisibles por el stride; `coset`
igual al stride total (producto de `strides` en CvT2/CvT3, `token_size` en ViT2D).

> **Subespacio del stride.** Con stride (2,2) en una red L×L, los momentos (0,0),
> (π,π), (π,0) y (0,π) comparten los mismos caracteres sobre la malla de tokens: la misma
> red sirve para los cuatro cambiando sólo la fase del coset. (π/2,0) queda fuera.

### Envoltorio Z2

Sin parámetros. Opciones: `trivial_Z2` (`true`: ψ(x)+ψ(−x), sector par; `false`:
ψ(x)−ψ(−x), sector impar) y `symm_modulus` (sólo simetriza el módulo; no da un
autoestado exacto). Supone espines ±1. El ED de sector Z2 exige `sz_total = 0`.

### Envoltorio Traslation

Opciones `irrep`, `use_anchor`, `save_memory`. **Actualmente no es funcional** (llama a
`Translations2D` con argumentos que ya no existen). Para sectores de momento use
`Sequivariant` con `coset`.

## 4.5 Etapas, carga y trasplante de pesos

### Evolución (`Hydra.arch_evol`)

Al final de cada eon del schedule (§5.1) `VMC_simulation.train` llama a
`hydra.arch_evol`, crea un nuevo `MCState` y trasplanta los pesos:

1. Si la etapa actual tiene `save_params: true`, sus parámetros se guardan en memoria para
   etapas futuras; la inmediatamente anterior siempre está disponible.
2. `load` de la nueva etapa:
   * ausente → `[[etapa_previa, "A", "A"]]` (trasplantar todo lo compatible);
   * `[]` → inicialización aleatoria.
3. `symm_wrapper` de la etapa: modifica opciones de envoltorios existentes, p. ej.
   `{"Z2": {"trivial_Z2": false}}`. Los cambios son **acumulativos** para las etapas
   siguientes.
4. `change_attr`: `{"Módulo": {"atributo": valor, "_count": k}}`; sin `_count` cambia
   todas las apariciones, con `_count: k` se saltan las k primeras. **No se aplica en
   `stage0`**.
5. Se guarda `storage["stage<k>"]` (no use nombres de módulo que empiecen por `stage`).

### Instrucciones `load`

`[[origen, código_viejo, código_nuevo], ...]`:

* `origen`: `"stageK"` (guardada con `save_params` o la inmediatamente anterior) o
  `"stageX<i>"` = parámetros leídos de `load_model.load_from[i]` (sólo en `stage0`).
* Códigos o `"A"`. Ejemplo: `[["stage0","00","0"], ["stage0","01","1"]]`.

**Trasplante** (`greedy_transplant`): compatibles = misma forma y mismo tipo
(real/complejo); la precisión puede diferir (se convierte al dtype del destino). Primero se
emparejan hojas con la misma ruta relativa; las sobrantes se asignan al primer hueco
compatible libre (esto puede emparejar arrays no relacionados: use códigos precisos si las
arquitecturas difieren). Lo que no se rellena queda con inicialización aleatoria y se lista
como «random init».

### Arranque desde otra simulación (`load_model`)

| Campo | Significado |
|---|---|
| `parameters` | Cargar parámetros (requiere `stage0.load`) |
| `sampler` | Cargar el estado del sampler (cadenas) |
| `checkpoint` | Paso de checkpoint (sólo con rutas `*_checkpoint.orbax`) |
| `load_from` | Lista de orígenes |

Cada origen puede ser:

| Origen | Qué se carga |
|---|---|
| JSON de resultados | `_artifacts.parameters` / `sampler_state` (el mejor estado de la simulación) |
| Carpeta | El `*results*.json` más reciente bajo ella; si no hay (la simulación murió o sigue en marcha), el `*_best.orbax` más reciente |
| Glob | La coincidencia más reciente |
| `…_best.orbax[/modo[/stage<k>]]` | El mejor estado guardado por el keeper; si faltan niveles: modo preferido `best > best_energy > best_vscore > always`, última etapa |
| `…_checkpoint.orbax` | El paso `checkpoint` |

Los checkpoints guardados con varias GPUs se cargan en cualquier número de dispositivos
(restauración como arrays numpy).

## 4.6 Catálogo de módulos simples

Convenciones: la entrada llega como `(B, L0, L1, 1)`; un backbone devuelve una malla
`(B, H′, W′, C)` (o tokens aplanados `(B, N′, C)` donde se indica); una cabeza devuelve
log ψ `(B, 1)`. `final_architecture`, cuando no es `null`, añade pooling + cadena Dense
(normalmente se deja `null` y se usa una cabeza aparte).

### ViT2D — transformer de visión con atención posicional factorizada

| Campo | Significado |
|---|---|
| `token_size` | Tamaño del parche (ty, tx); debe dividir la red |
| `embedding_d` | Dimensión del embedding (divisible por `n_heads`) |
| `n_heads` | Cabezas; atención circulante aprendible independiente de la entrada |
| `n_blocks` | Bloques `x += Attn(LN(x)); x = MLP(x) + x` |
| `n_ffn_layers` | Capas Dense→LN→swish de la FFN |
| `final_architecture` | `null` |

Salida `(B, Ht, Wt, embedding_d)`; equivariante bajo traslaciones por el token. Soporta
`network_dtype`.

### CvT3 — transformer convolucional (backbone principal)

Por etapa: embedding convolucional `Conv(channels[i], kernel[i], strides[i], CIRCULAR)` →
`n_CP_blocks[i]` bloques → LayerNorm. Q, K, V: convolución *depthwise* 3×3 circular → LN →
1×1. Atención softmax multi-cabeza.

| Campo | Defecto | Significado |
|---|---|---|
| `n_CP_blocks` | — | Bloques por etapa (su longitud = nº de etapas) |
| `channels` | — | Canales por etapa (divisibles por `attn_heads`) |
| `attn_heads` | — | Cabezas por etapa |
| `kernel` | — | Kernel por etapa; **sólo afecta al embedding** (Q/K/V usan 3×3) |
| `strides` | — | Stride por etapa; el producto es el `coset` adecuado |
| `ffn_act` | `"swish"` | `swish, gelu, relu, elu, tanh` |
| `ffn_factor` | `2` | Anchura oculta de la FFN = factor·C |
| `ffn_out` | `"act_ln"` | `"act_ln"` (original: la rama residual es swish(LN(·)) ≥ −0.28) o `"linear"` (FFN estándar) |
| `norm` | `"mixed"` | `"mixed"` (original), `"pre"` o `"post"` |
| `attn_out_proj` | `false` | Proyección W_O tras las cabezas |
| `final_architecture` | `null` | — |

Los valores por defecto reproducen bit a bit la red original (checkpoints compatibles).
Las combinaciones de opciones están cubiertas por tests de equivariancia. Soporta
`network_dtype`. Experiencia práctica: en 6x6/8x8 el kernel 3x3 en el embedding entrena
mucho mejor que 5x5 (D4: −126.06 frente a −40.8 en 200 épocas en 8x8).

### Otros backbones

| Módulo | Resumen | Notas |
|---|---|---|
| `CvT` | CvT con kernel común (embedding y Q/K/V), stride 1, etapas terminadas en log_cosh | No sigue `network_dtype` |
| `CvT2` | Como CvT con kernel y stride por etapa | — |
| `CvTaps` | CvT con *adaptive polyphase sampling* en convs con stride | Salida aplanada `(B,N′,C)`: no sirve con Sequivariant |
| `CvTexp` | Varios kernels por bloque de atención (experimental) | Salida aplanada |
| `VViT` | ViT con cabeza propia | Probablemente roto bajo NNWrapper |
| `CNN` | Bloques Conv→LN→swish | Salida aplanada |
| `CNNLiang` | Red de Liang et al. 2021 | Equivariante completa sólo con `window_pooling=1` |
| `MLP`, `CMLP` | Perceptrones real y complejo | `MLP.only_phase=true` siempre falla |
| `Jastrow_wrap`, `Factorized`, `FactorMod` | Ansätze simples | Pensados para entrada plana |
| `CNNPh`, `EDPPh`, `CNNClsf`, `CNNbinClsf`, `CNNSzabo` | Redes de fase | Uso experimental |
| `MarshallSign`, `CarreteSign` | Signo de Marshall / ancla de traslación | — |
| `ARViT` | Transformer autorregresivo (muestreo directo) | Activa `ARNNWrapper`; sin envoltorios de simetría |

### Cabezas

| Módulo | Campos | Salida |
|---|---|---|
| `OutputHead` | `d_model`, `only_phase=false` | Suma sobre tokens → LN → ramas real/imag Dense+LN → Σ log_cosh(z): `(B,1)` complejo. **Cabeza estándar con Sequivariant.** Sigue `network_dtype`. |
| `DeepOutputHead` | `d_model` (lista), `activations_real/imag` | Como OutputHead con cadenas Dense+LN (las activaciones no llegan a aplicarse) |
| `ComplexHead` | `d_model`, `only_phase` | Suma → LN → Dense complejo → Σ log_cosh |
| `SzaboOutput` | — | Para salidas aplanadas `(B,N,C)` |
| `Sum`, `Mean`, `FinalFF` | — | Reductores |

## 4.7 Precisión numérica de la red

Ver §5.5. Resumen: `network_dtype: "float32"` ejecuta ViT2D, CvT3 y sus cabezas en float32
con parámetros, Jacobiano y SR en float64 (unas 4 veces más rápido); `matmul_precision`
debe quedarse en `"highest"`.

## 4.8 Añadir un módulo nuevo

1. Escribir un `nn.Module` de flax en `Hydra/NN/NN/SingleModules/MiRed.py`:
   * los campos del dataclass son los hiperparámetros (las listas JSON llegan como tuplas);
   * evite nombres de campo con las subcadenas reservadas (§4.2);
   * entrada `(B, L0, L1, 1)`; para usarlo bajo `Sequivariant`, devuelva una malla
     equivariante `(B,H,W,C)` con padding circular;
   * para soportar float32, use un global `DTYPE` del módulo para `dtype`/`param_dtype`,
     convierta la entrada con `x.astype(DTYPE)` y registre el módulo en
     `precision._MODULES`;
   * no use convoluciones agrupadas (falla `jax.jacrev` bajo la malla de NetKet): use
     `toolbox.DepthwiseConv`.
2. Exportarlo en `Hydra/NN/NN/SingleModules/__init__.py`.
3. En `Hydra/NN/__init__.py`: importarlo, añadirlo a `__all_single__` y a
   `REGISTRY_SINGLE`; si necesita la forma de la red o el espacio de Hilbert, añadirlo a
   `LATTICE_SIZE` o `HILBERT`.
4. Añadir `storage["MiRed"]` en config3 y usarlo en una plantilla como `{"MiRed": {}}`.
5. Verificar con `pytests/test_equivariance` (`_trasl_equiv_check.traslation_equivariant`),
   `pytests/test_hydra` y `pytests/test_diagram`.

Para un **módulo de flujo** nuevo: `REGISTRY_FLUX`, `__all_flux__`,
`flux_submodule_dict` (nombre exacto del campo hijo), rama en `get_submodules`, la etiqueta
en `submodule_tags` y, si la aridad es variable, `undefined_submodule_number`. Para un
**envoltorio de simetría**: `REGISTRY_SYMM`, `__all_symm__`, `symm_submodule_dict`
(`"<X>Wrap"`), su entrada en `config3.symm_wrapper` y, si fija un sector, ampliar
`exact_diagonalization._sector`.
