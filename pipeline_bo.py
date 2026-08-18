"""
pipeline_bo_generico.py
========================
Otimizacao bayesiana de geometria de antena (HFSS) para uma grade de pares
(f_res_alvo, er_alvo). O otimizador busca apenas
sobre as variaveis geometricas; a permissividade do material e fixada
no valor alvo antes da otimizacao comecar.

Authors : See CITATION.cff
Revision: 2026-08-14
"""
import os
import time
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import qmc, norm
from scipy.optimize import differential_evolution
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel, RBF
from sklearn.preprocessing import StandardScaler

import copy

# =============================================================================
# CONFIGURACAO DA ANTENA — UNICO BLOCO A EDITAR PARA CADA PROJETO NOVO
# =============================================================================
ANTENNA_CONFIG = {
    # Set HFSS_PROJECT_PATH to the local .aedt file before running.
    'project_path': os.environ.get('HFSS_PROJECT_PATH', ''),
    # Results are written to RESULTS_DIR, or to ./results by default.
    'base_dir':     os.environ.get('RESULTS_DIR', 'results'),

    'freq_unit':         'THz',
    'freq_sweep':        (8.5, 12.5),
    'freq_sweep_points': 801,
    'freq_valid_range':  (8.7, 12.3),

    's11_threshold_db': -15.0,
    's11_reject_db':    -20.0,

    'material_name': 'rutile',

    # nome exato da variavel no HFSS -> (minimo, maximo, unidade)
    'geometric_vars': {
        'a':      (8.6, 9.6, 'um'),
        'b':      (8.2, 9.2, 'um'),
        'd':      (6.4, 7.4, 'um'),
        'lstrip': (13.5, 14.5, 'um'),
        'wstrip': (1.325, 1.525, 'um'),
        'lslot':  (0.1, 0.9, 'um'),
        'wslot':  (5.8, 6.6, 'um'),
    },
}

N_INIT           = 12
N_ITER_BO        = 30
EI_MIN           = 1e-6
PENALIDADE       = 5.0
SEMENTE          = 42
CASAS_FABRICACAO = 3

W_FREQ, W_S11, W_RE_Z, W_IM_Z, W_VSWR = 0.60, 0.20, 0.10, 0.10, 0.00

# =============================================================================
# GRADE DE OTIMIZACAO — EDITE SOMENTE ESTES DOIS VETORES
# =============================================================================
GRADE_FREQUENCIAS = [10.0, 11.0]
GRADE_ER         = [8.0, 9.0, 10.0]

log = logging.getLogger('pipeline_bo_generico')
if not log.handlers:
    log.addHandler(logging.StreamHandler())
    log.setLevel(logging.INFO)


# =============================================================================
# VALIDACAO DE CONFIG
# =============================================================================
def _validar_config(config: dict) -> None:
    obrigatorias = ['project_path', 'base_dir', 'freq_unit', 'freq_sweep',
                     'freq_sweep_points', 'freq_valid_range', 's11_threshold_db',
                     's11_reject_db', 'material_name', 'geometric_vars']
    faltando = [k for k in obrigatorias if k not in config]
    if faltando:
        raise ValueError(f"ANTENNA_CONFIG incompleto — faltam as chaves: {faltando}")

    if not config.get('project_path'):
        raise ValueError(
            "project_path nao definido. Defina a variavel de ambiente "
            "HFSS_PROJECT_PATH apontando para o arquivo .aedt local."
        )

    if not os.path.isfile(config['project_path']):
        raise FileNotFoundError(
            f"Projeto HFSS nao encontrado: {config['project_path']}. "
            "Defina HFSS_PROJECT_PATH para um arquivo .aedt local."
        )

    if not config.get('material_name'):
        raise ValueError("material_name deve ser definido — o Er alvo precisa de um material HFSS associado.")

    if config['freq_sweep'][0] >= config['freq_sweep'][1]:
        raise ValueError("freq_sweep deve ser (inicio, fim) com inicio < fim.")

    if not (config['freq_sweep'][0] <= config['freq_valid_range'][0]
            and config['freq_valid_range'][1] <= config['freq_sweep'][1]):
        raise ValueError("freq_valid_range deve estar contido em freq_sweep.")

    if config['s11_reject_db'] >= config['s11_threshold_db']:
        raise ValueError(
            "s11_reject_db deve ser mais negativo que s11_threshold_db. "
            f"Recebido: threshold={config['s11_threshold_db']}, reject={config['s11_reject_db']}.")

    if not config['geometric_vars']:
        raise ValueError("geometric_vars esta vazio — a busca precisa de pelo menos 1 variavel.")

    for nome, spec in config['geometric_vars'].items():
        if len(spec) != 3:
            raise ValueError(f"geometric_vars['{nome}'] deve ser (min, max, unidade).")
        mn, mx, _unid = spec
        if mn >= mx:
            raise ValueError(f"geometric_vars['{nome}']: minimo ({mn}) deve ser < maximo ({mx}).")


# =============================================================================
# UTILITARIOS HFSS
# =============================================================================
def configurar_log(log_path: str, nome_logger: str = 'pipeline_bo_generico') -> logging.Logger:
    logger = logging.getLogger(nome_logger)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formato = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

    fh = logging.FileHandler(log_path, encoding='utf-8')
    fh.setFormatter(formato)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formato)
    logger.addHandler(sh)

    logger.propagate = False
    return logger


def conectar_hfss(project_path: str, max_tentativas: int = 3):
    from pyaedt import Hfss
    for tentativa in range(1, max_tentativas + 1):
        try:
            return Hfss(projectname=project_path, non_graphical=False,
                        new_desktop_session=True, close_on_exit=False)
        except Exception as e:
            log.warning(f"Tentativa {tentativa} de conexao falhou: {e}")
            time.sleep(5)
    raise RuntimeError("Nao foi possivel conectar ao HFSS apos 3 tentativas.")


def encerrar_hfss(hfss) -> None:
    """Libera PyAEDT e fecha os projetos e o Desktop do HFSS.

    Compatibilidade: algumas versoes do PyAEDT usam ``close_desktop`` e
    outras usam ``close_on_exit`` em ``release_desktop``.
    """
    if hfss is None:
        return

    try:
        hfss.release_desktop(close_projects=True, close_desktop=True)
        log.info("Sessao HFSS/PyAEDT encerrada: projetos e Desktop fechados.")
        return
    except TypeError:
        # API de versoes que usam close_on_exit no release_desktop.
        try:
            hfss.release_desktop(close_projects=True, close_on_exit=True)
            log.info("Sessao HFSS/PyAEDT encerrada: projetos e Desktop fechados.")
            return
        except Exception as e:
            log.warning(f"release_desktop falhou na assinatura alternativa: {e}")
    except Exception as e:
        log.warning(f"release_desktop falhou: {e}")

    # Ultimo recurso: algumas versoes expõem close_desktop() diretamente no Hfss.
    try:
        hfss.close_desktop()
        log.info("Desktop HFSS encerrado pelo metodo close_desktop().")
    except Exception as e:
        log.error(f"Nao foi possivel fechar o Desktop HFSS: {e}", exc_info=True)


def _deletar_reports(oModule, nomes: list[str]) -> None:
    for nome in nomes:
        try:
            oModule.DeleteReports([nome])
        except Exception as e:
            log.debug(f"DeleteReports('{nome}') ignorado: {e}")


def _vswr_from_s11_db(s11_db: float) -> float:
    s11_lin = 10 ** (s11_db / 20.0)
    s11_lin = min(abs(s11_lin), 0.9999)
    return (1 + s11_lin) / (1 - s11_lin)


def configurar_hfss_simulacao(hfss, config: dict):
    for s in hfss.setup_names:
        hfss.delete_setup(s)

    freq_centro = sum(config['freq_sweep']) / 2
    setup = hfss.create_setup("Setup1")
    setup.props["Frequency"]         = f"{freq_centro}{config['freq_unit']}"
    setup.props["MaxDeltaS"]         = 0.01
    setup.props["MaximumPasses"]     = 12
    setup.props["MinimumPasses"]     = 2
    setup.props["PercentRefinement"] = 30
    setup.update()

    for sw in hfss.get_sweeps("Setup1"):
        try:
            hfss.delete_sweep("Setup1", sw)
        except Exception as e:
            log.debug(f"delete_sweep('{sw}') ignorado: {e}")

    hfss.create_linear_count_sweep(
        setup="Setup1", units=config['freq_unit'],
        start_frequency=config['freq_sweep'][0],
        stop_frequency=config['freq_sweep'][1],
        num_of_freq_points=config['freq_sweep_points'],
        name="Sweep_dataset", sweep_type="Interpolating", save_fields=False
    )
    return hfss.get_sweeps("Setup1")[0]


def exportar_s11_simulacao(oModule, sweep_nome: str, csv_path: str,
                            max_tentativas: int = 3) -> None:
    for tentativa in range(1, max_tentativas + 1):
        try:
            _deletar_reports(oModule, ["S11_data"])
            time.sleep(2)
            oModule.CreateReport(
                "S11_data", "Modal Solution Data", "Rectangular Plot",
                f"Setup1 : {sweep_nome}", ["Domain:=", "Sweep"],
                ["Freq:=", ["All"]],
                ["X Component:=", "Freq", "Y Component:=", ["dB(S(1,1))"]], []
            )
            oModule.ExportToFile("S11_data", csv_path, False)
            if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
                return
            raise FileNotFoundError("CSV S11 nao gerado ou vazio.")
        except Exception as e:
            log.warning(f"Exportacao S11 tentativa {tentativa} falhou: {e}")
            time.sleep(3)
    raise RuntimeError("Falha na exportacao S11 apos 3 tentativas.")


def exportar_z11_simulacao(oModule, sweep_nome: str, csv_path: str,
                            max_tentativas: int = 3) -> None:
    for tentativa in range(1, max_tentativas + 1):
        try:
            _deletar_reports(oModule, ["Z11_data"])
            time.sleep(2)
            oModule.CreateReport(
                "Z11_data", "Modal Solution Data", "Rectangular Plot",
                f"Setup1 : {sweep_nome}", ["Domain:=", "Sweep"],
                ["Freq:=", ["All"]],
                ["X Component:=", "Freq",
                 "Y Component:=", ["re(Z(1,1))", "im(Z(1,1))"]], []
            )
            oModule.ExportToFile("Z11_data", csv_path, False)
            if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
                return
            raise FileNotFoundError("CSV Z11 nao gerado ou vazio.")
        except Exception as e:
            log.warning(f"Exportacao Z11 tentativa {tentativa} falhou: {e}")
            time.sleep(3)
    raise RuntimeError("Falha na exportacao Z11 apos 3 tentativas.")


def processar_s11(csv_path: str, config: dict):
    df = pd.read_csv(csv_path, skiprows=1, header=None, names=['freq', 's11_db'])
    if len(df) < 10:
        return None, None, None, "CSV_INVALIDO"

    abaixo = df['s11_db'] <= config['s11_threshold_db']
    regioes, dentro, inicio = [], False, 0

    for j in range(len(df)):
        if abaixo.iloc[j] and not dentro:
            inicio, dentro = j, True
        elif not abaixo.iloc[j] and dentro:
            fim = j - 1
            idx_min = df['s11_db'].iloc[inicio:fim + 1].idxmin()
            regioes.append({'f_res': df['freq'].iloc[idx_min],
                             's11_min': df['s11_db'].iloc[idx_min],
                             'bw': round(df['freq'].iloc[fim] - df['freq'].iloc[inicio], 6)})
            dentro = False

    if dentro:
        fim = len(df) - 1
        idx_min = df['s11_db'].iloc[inicio:fim + 1].idxmin()
        regioes.append({'f_res': df['freq'].iloc[idx_min],
                         's11_min': df['s11_db'].iloc[idx_min],
                         'bw': round(df['freq'].iloc[fim] - df['freq'].iloc[inicio], 6)})

    if len(regioes) != 1:
        return None, None, None, f"MULTIPLAS ({len(regioes)})"

    r = regioes[0]
    f_res, s11_res, bw = r['f_res'], r['s11_min'], r['bw']
    f_min, f_max = config['freq_valid_range']

    if not (f_min <= f_res <= f_max):
        return None, None, None, "LIMITE_SWEEP"
    if s11_res > config['s11_reject_db']:
        return None, None, None, "S11_FRACO"
    if bw <= 0:
        return None, None, None, "BW_ZERO"

    idx_min_global = df['s11_db'].idxmin()
    trecho = df['s11_db'].iloc[:idx_min_global].values
    if len(trecho) > 1:
        subidas = np.diff(trecho)
        if len(subidas[subidas > 1.0]) > 0:
            return None, None, None, "NAO_MONOTONICA"

    return round(f_res, 6), round(s11_res, 4), round(bw, 6), "OK"


def processar_z11_vswr(f_res, csv_z11, csv_s11):
    try:
        df_z   = pd.read_csv(csv_z11, skiprows=1, header=None, names=['freq', 're_z11', 'im_z11'])
        idx_z  = (df_z['freq'] - f_res).abs().idxmin()
        re_z11 = round(float(df_z['re_z11'].iloc[idx_z]), 4)
        im_z11 = round(float(df_z['im_z11'].iloc[idx_z]), 4)

        df_s   = pd.read_csv(csv_s11, skiprows=1, header=None, names=['freq', 's11_db'])
        idx_s  = (df_s['freq'] - f_res).abs().idxmin()
        vswr   = round(_vswr_from_s11_db(df_s['s11_db'].iloc[idx_s]), 4)
        return re_z11, im_z11, vswr
    except Exception as e:
        log.warning(f"Erro ao processar Z11/VSWR: {e}")
        return np.nan, np.nan, np.nan


# =============================================================================
# ESPACO DE BUSCA (apenas variaveis geometricas — Er e fixado externamente)
# =============================================================================
def montar_espaco_busca(config: dict):
    nomes = list(config['geometric_vars'].keys())
    bounds = np.array([[mn, mx] for mn, mx, _unid in config['geometric_vars'].values()], dtype=float)
    return nomes, bounds


def simular_ponto(hfss, oModule, sweep_nome, config: dict, valores: dict,
                   retornar_curva: bool = False, csv_prefixo: str = 'bo_temp'):
    for nome, (_, _, unidade) in config['geometric_vars'].items():
        hfss[nome] = f"{valores[nome]}{unidade}"

    hfss.analyze_setup("Setup1")

    csv_s11 = os.path.join(config['base_dir'], f'{csv_prefixo}_s11.csv')
    csv_z11 = os.path.join(config['base_dir'], f'{csv_prefixo}_z11.csv')
    exportar_s11_simulacao(oModule, sweep_nome, csv_s11)
    exportar_z11_simulacao(oModule, sweep_nome, csv_z11)

    f_res, s11_min, bw, qualidade = processar_s11(csv_s11, config)
    if qualidade != "OK":
        return None

    re_z11, im_z11, vswr = processar_z11_vswr(f_res, csv_z11, csv_s11)
    if any(np.isnan(v) for v in (re_z11, im_z11, vswr)):
        log.warning("Z11/VSWR nao computavel para este ponto — rejeitando.")
        return None

    resultado = dict(f_res=f_res, s11_min=s11_min, bw=bw, re_z11=re_z11, im_z11=im_z11, vswr=vswr)

    if retornar_curva:
        resultado['_curva_s11'] = pd.read_csv(csv_s11, skiprows=1, header=None, names=['freq', 's11_db'])
    return resultado


# =============================================================================
# CUSTO, AMOSTRAGEM INICIAL, EI E CLASSIFICADOR DE VIABILIDADE
# =============================================================================
def calcular_custo(f_res, s11_min, re_z11, im_z11, vswr, f_res_alvo,
                    freq_norm, s11_ideal=-40.0, s11_norm=20.0) -> float:
    erro_freq = abs(f_res - f_res_alvo) / freq_norm
    erro_s11  = max((s11_min - s11_ideal) / s11_norm, 0.0)
    erro_re_z = abs(re_z11 - 50.0) / 50.0
    erro_im_z = abs(im_z11) / 20.0
    erro_vswr = abs(vswr - 1.0)
    return (W_FREQ * erro_freq + W_S11 * erro_s11 +
            W_RE_Z * erro_re_z + W_IM_Z * erro_im_z + W_VSWR * erro_vswr)


def amostragem_inicial(n_init: int, bounds_arr: np.ndarray, seed: int = SEMENTE):
    sampler      = qmc.LatinHypercube(d=bounds_arr.shape[0], seed=seed)
    amostra_unit = sampler.random(n=n_init)
    return qmc.scale(amostra_unit, bounds_arr[:, 0], bounds_arr[:, 1])


def expected_improvement(X, gp, y_best, xi=0.01):
    mu, sigma  = gp.predict(X, return_std=True)
    sigma_safe = np.maximum(sigma, 1e-9)
    imp = y_best - mu - xi
    z   = imp / sigma_safe
    ei  = imp * norm.cdf(z) + sigma_safe * norm.pdf(z)
    return np.where(sigma < 1e-9, 0.0, ei)


def _treinar_classificador_viabilidade(X_sc: np.ndarray, viavel: list):
    if len(set(viavel)) < 2:
        return None
    try:
        clf = GaussianProcessClassifier(kernel=RBF(length_scale=1.0),
                                         random_state=SEMENTE, n_restarts_optimizer=2)
        clf.fit(X_sc, viavel)
        return clf
    except Exception as e:
        log.warning(f"Classificador de viabilidade nao pode ser treinado nesta iteracao: {e}")
        return None


def acquisition_ei_viabilidade(X, gp_custo, y_best, clf_viab=None, xi=0.01):
    ei = expected_improvement(X, gp_custo, y_best, xi=xi)
    if clf_viab is not None:
        p_viavel = clf_viab.predict_proba(X)[:, 1]
        ei = ei * p_viavel
    return ei


def proximo_ponto(gp_custo, y_best, scaler_X, bounds_arr, seed,
                   clf_viab=None, xi=0.01, n_restarts: int = 4):
    def neg_aquisicao(x):
        x_sc = scaler_X.transform(x.reshape(1, -1))
        return -acquisition_ei_viabilidade(x_sc, gp_custo, y_best, clf_viab, xi)[0]

    melhor_x, melhor_val = None, np.inf
    for r in range(n_restarts):
        res = differential_evolution(neg_aquisicao, bounds_arr, seed=seed + r * 997,
                                      maxiter=300, tol=1e-8, polish=True)
        if res.fun < melhor_val:
            melhor_val, melhor_x = res.fun, res.x
    return melhor_x, -melhor_val


def _diagnosticar_orcamento(nomes: list, n_init: int, n_iter: int) -> None:
    d = len(nomes)
    pts_por_dim = n_init / d
    if pts_por_dim < 3:
        log.warning(
            f"[Diagnostico de orcamento] {d} dimensoes, n_init={n_init} "
            f"({pts_por_dim:.1f} pontos/dimensao). Regra pratica: use pelo "
            f"menos ~4x(d+1) = {4 * (d + 1)} pontos iniciais.")


# =============================================================================
# ARREDONDAMENTO PARA FABRICACAO
# =============================================================================
def arredondar_fabricacao(valores: dict, casas: int = CASAS_FABRICACAO) -> dict:
    return {k: round(float(v), casas) for k, v in valores.items()}


def tabela_resultado_final(valores: dict, resultado: dict, config: dict, er_alvo: float) -> pd.DataFrame:
    linha = dict(valores)
    linha['er'] = er_alvo
    linha[f"f_res_{config['freq_unit']}"] = round(resultado['f_res'], 4)
    linha['S11_min_dB']  = round(resultado['s11_min'], 2)
    linha['Re_Z11_ohm']  = round(resultado['re_z11'], 2)
    linha['Im_Z11_ohm']  = round(resultado['im_z11'], 2)
    linha['VSWR']        = round(resultado['vswr'], 3)
    linha[f"BW_{config['freq_unit']}"] = round(resultado['bw'], 4)
    return pd.DataFrame([linha])


# =============================================================================
# GRAFICOS PARA O ARTIGO (qualidade de publicacao, 300 dpi)
# =============================================================================
def _aplicar_estilo_artigo():
    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'serif',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })


def plot_convergencia(df_result: pd.DataFrame, config: dict, save_path: str):
    _aplicar_estilo_artigo()
    df = df_result.sort_values('iteracao').reset_index(drop=True)
    custo_min_acumulado = df['custo'].cummin()

    fig, ax = plt.subplots(figsize=(7, 5))
    mask_init = df['fase'] == 'init'
    mask_bo   = df['fase'] == 'bo'

    ax.scatter(df.loc[mask_init, 'iteracao'], df.loc[mask_init, 'custo'],
               color='#888888', s=32, label='Initial LHS samples', zorder=3)
    ax.scatter(df.loc[mask_bo, 'iteracao'], df.loc[mask_bo, 'custo'],
               color='#1f77b4', s=32, label='BO-selected samples (EI)', zorder=3)
    ax.plot(df['iteracao'], custo_min_acumulado, color='#d62728', lw=2,
            label='Best cost so far', zorder=2)

    ax.set_yscale('log')
    ax.set_xlabel('HFSS evaluation index')
    ax.set_ylabel('Combined cost (log scale)')
    ax.set_title('Bayesian Optimization Convergence')
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_s11_final(df_curva_s11: pd.DataFrame, f_res_alvo: float,
                    resultado_fab: dict, config: dict, save_path: str, er_alvo: float):
    _aplicar_estilo_artigo()
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(df_curva_s11['freq'], df_curva_s11['s11_db'], color='#1f77b4', lw=1.8)
    ax.axvline(f_res_alvo, color='#d62728', ls='--', lw=1.2,
               label=f"Target: {f_res_alvo} {config['freq_unit']}")
    ax.axhline(-10, color='gray', ls=':', lw=1, label='-10 dB reference')
    ax.scatter([resultado_fab['f_res']], [resultado_fab['s11_min']],
               color='#2ca02c', s=50, zorder=5,
               label=(f"Fabrication design: {resultado_fab['f_res']:.4f} "
                      f"{config['freq_unit']}, {resultado_fab['s11_min']:.2f} dB"))

    linhas_info = [
        f"$\\varepsilon_r$ = {er_alvo:.3f}",
        f"$f_{{res}}$ = {resultado_fab['f_res']:.4f} {config['freq_unit']}",
        f"$S_{{11,min}}$ = {resultado_fab['s11_min']:.2f} dB",
        f"VSWR = {resultado_fab['vswr']:.3f}",
    ]
    ax.text(0.02, 0.03, "\n".join(linhas_info), transform=ax.transAxes,
            fontsize=9, va='bottom', ha='left',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='#888888'))

    ax.set_xlabel(f"Frequency ({config['freq_unit']})")
    ax.set_ylabel(r'$S_{11}$ (dB)')
    ax.set_title(f"Reflection Coefficient of the Optimized Design ($\\varepsilon_r$ = {er_alvo:.3f})")
    ax.legend(frameon=False, fontsize=8.5, loc='upper right')
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_exploracao_espaco(df_result: pd.DataFrame, nomes: list,
                            bounds_arr: np.ndarray, save_path: str):
    _aplicar_estilo_artigo()
    df_validos    = df_result[~df_result.get('rejeitado', False)].copy()
    df_rejeitados = df_result[df_result.get('rejeitado', False)].copy()
    n_vars = len(nomes)

    def _normalizar(df):
        normed = np.zeros((len(df), n_vars))
        for i, nome in enumerate(nomes):
            mn, mx = bounds_arr[i]
            normed[:, i] = (df[nome].values - mn) / (mx - mn)
        return normed

    fig, ax = plt.subplots(figsize=(1.6 * n_vars + 2, 5))
    xs = np.arange(n_vars)

    if len(df_rejeitados) > 0:
        normed_rej = _normalizar(df_rejeitados)
        for i in range(len(df_rejeitados)):
            ax.plot(xs, normed_rej[i], color='#999999', alpha=0.5, lw=1.0, ls='--', zorder=1)
        ax.plot([], [], color='#999999', ls='--', lw=1.5, label=f'Rejected ({len(df_rejeitados)})')

    if len(df_validos) > 0:
        normed = _normalizar(df_validos)
        custo = df_validos['custo'].values
        vmin, vmax = custo.min(), custo.max()
        norm_c = (custo - vmin) / (vmax - vmin + 1e-12)
        cmap = plt.get_cmap('viridis_r')

        for i in range(len(df_validos)):
            ax.plot(xs, normed[i], color=cmap(norm_c[i]), alpha=0.6, lw=1.2, zorder=2)

        idx_best = int(np.argmin(custo))
        ax.plot(xs, normed[idx_best], color='red', lw=2.5, zorder=5, label='Best design')

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label='Combined cost (valid points only)')

    ax.set_xticks(xs)
    ax.set_xticklabels(nomes)
    ax.set_ylabel('Normalized value within search bounds')
    ax.set_title('Parameter Space Exploration (colored by cost)')
    ax.legend(frameon=False, fontsize=9)

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_ei_decay(df_result: pd.DataFrame, save_path: str):
    df_bo = df_result[(df_result['fase'] == 'bo') & df_result['ei'].notna()]
    if df_bo.empty:
        return False
    _aplicar_estilo_artigo()
    df_bo = df_bo.sort_values('iteracao')

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(df_bo['iteracao'], df_bo['ei'], marker='o', color='#9467bd', lw=1.5)
    ax.axhline(EI_MIN, color='gray', ls='--', lw=1, label=f'Stopping threshold ({EI_MIN:.0e})')
    ax.set_yscale('log')
    ax.set_xlabel('HFSS evaluation index')
    ax.set_ylabel('Max acquisition — EI x P(valid) (log scale)')
    ax.set_title('Acquisition Function Decay')
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return True


def gerar_figuras_artigo(df_result: pd.DataFrame, df_curva_s11: pd.DataFrame,
                          nomes: list, bounds_arr: np.ndarray, config: dict,
                          f_res_alvo: float, resultado_fab: dict, er_alvo: float) -> dict:
    base = config['base_dir']
    caminhos = {
        'convergencia': os.path.join(base, 'fig1_convergencia.png'),
        's11':          os.path.join(base, 'fig2_s11_final.png'),
        'exploracao':   os.path.join(base, 'fig3_exploracao_espaco.png'),
        'ei_decay':     os.path.join(base, 'fig4_ei_decay.png'),
    }
    plot_convergencia(df_result, config, caminhos['convergencia'])
    plot_s11_final(df_curva_s11, f_res_alvo, resultado_fab, config, caminhos['s11'], er_alvo)
    plot_exploracao_espaco(df_result, nomes, bounds_arr, caminhos['exploracao'])
    if not plot_ei_decay(df_result, caminhos['ei_decay']):
        caminhos.pop('ei_decay')
    return caminhos


def imprimir_resumo_final(tabela: pd.DataFrame, caminhos_fig: dict,
                           resultados_path: str, tabela_path: str) -> None:
    linha = tabela.iloc[0].to_dict()
    L = 60
    log.info("\n" + "=" * L)
    log.info("RESUMO FINAL - DESIGN OTIMIZADO")
    log.info("=" * L)
    for chave, valor in linha.items():
        if isinstance(valor, float):
            log.info(f"  {chave:<20s}: {valor:.4f}")
        else:
            log.info(f"  {chave:<20s}: {valor}")
    log.info("-" * L)
    log.info(f"  Figuras geradas ({len(caminhos_fig)}):")
    for nome_fig, caminho in caminhos_fig.items():
        log.info(f"    - {nome_fig}: {caminho}")
    log.info(f"  Tabela final       : {tabela_path}")
    log.info(f"  Historico completo : {resultados_path}")
    log.info("=" * L)


# =============================================================================
# REMODELAGEM ADAPTATIVA DOS BOUNDS GEOMETRICOS
# =============================================================================
# A remodelagem foi desenhada para evitar o efeito "sanfona" observado quando
# uma variável é alterada isoladamente e, na rodada seguinte, outra variável
# passa a ocupar a borda. A decisão agora usa a distribuição dos melhores pontos
# e expande simultaneamente todas as variáveis que apresentarem pressão persistente.
#
# O fluxo por par (f_res_alvo, er_alvo) fica:
#   BO 1 -> se necessario, expansao coordenada -> BO 2
#       -> se necessario, expansao coordenada -> BO 3 -> encerra.
# Assim, ha no maximo 3 otimizacoes completas por par.
MARGEM_BORDA                    = 0.03   # 3% finais do intervalo
FATOR_EXPANSAO_PRINCIPAIS       = 0.18   # +18% na largura quando a/b/d pressionam a borda
FATOR_EXPANSAO_SECUNDARIAS      = 0.10   # +10% para as demais variaveis pressionadas
MAX_REMODELAGENS                = 2     # 2 expansoes apos a BO inicial -> no max. 3 BOs
TOP_K_PERSISTENCIA              = 5     # olha para os K melhores pontos validos
MIN_FRACAO_BORDA_PERSISTENTE    = 0.60  # pelo menos 3 de 5 na mesma borda
VARS_PRIORITARIAS               = ('a', 'b', 'd')


def _detectar_pressao_na_borda(df_result: pd.DataFrame, config: dict,
                               margem: float = MARGEM_BORDA,
                               top_k: int = TOP_K_PERSISTENCIA,
                               min_fracao: float = MIN_FRACAO_BORDA_PERSISTENTE) -> dict:
    """Detecta pressao persistente nos limites usando os melhores pontos validos.

    A remodelagem nao e disparada apenas pelo melhor ponto. Para cada variavel,
    exige-se que o melhor ponto esteja na borda e que uma fracao minima dos
    ``top_k`` melhores pontos validos esteja concentrada na mesma borda.

    Retorna um dicionario por variavel com:
        {'lado': 'inferior'|'superior', 'fracao': float, 'n_borda': int,
         'top_k': int, 'pos_melhor': float}
    somente para variaveis que justificam expansao.
    """
    if df_result is None or df_result.empty:
        return {}

    if 'rejeitado' in df_result.columns:
        df_validos = df_result[~df_result['rejeitado'].fillna(False)].copy()
    else:
        df_validos = df_result.copy()

    if df_validos.empty:
        return {}

    # Os resultados ja costumam vir ordenados por custo, mas ordenamos novamente
    # para garantir que a decisao nao dependa da ordem de escrita do CSV.
    if 'custo' in df_validos.columns:
        df_validos = df_validos.sort_values('custo', ascending=True)

    melhores = df_validos.head(max(1, int(top_k)))
    n_top = len(melhores)
    pressao = {}

    for nome, (mn, mx, _unid) in config['geometric_vars'].items():
        largura = mx - mn
        if largura <= 0 or nome not in melhores.columns:
            continue

        posicoes = (melhores[nome].astype(float) - mn) / largura
        pos_melhor = float(posicoes.iloc[0])

        n_inf = int((posicoes <= margem).sum())
        n_sup = int((posicoes >= 1.0 - margem).sum())
        frac_inf = n_inf / n_top
        frac_sup = n_sup / n_top

        # A melhor solucao precisa estar na mesma borda que concentra os top-k.
        if pos_melhor <= margem and frac_inf >= min_fracao:
            pressao[nome] = {
                'lado': 'inferior',
                'fracao': frac_inf,
                'n_borda': n_inf,
                'top_k': n_top,
                'pos_melhor': pos_melhor,
            }
        elif pos_melhor >= 1.0 - margem and frac_sup >= min_fracao:
            pressao[nome] = {
                'lado': 'superior',
                'fracao': frac_sup,
                'n_borda': n_sup,
                'top_k': n_top,
                'pos_melhor': pos_melhor,
            }

    return pressao


def _remodelar_bounds_coordenado(config: dict, pressao: dict) -> dict:
    """Expande simultaneamente todas as variaveis com pressao persistente.

    A expansao ocorre na direcao indicada pela borda e preserva o centro do
    intervalo atual. Nao existe recentragem no melhor ponto: isso evita que
    uma mudanca em ``a`` desloque artificialmente o espaco e provoque uma
    sucessao a -> d -> b -> a nas rodadas seguintes.

    ``a``, ``b`` e ``d`` recebem expansao maior por dominarem a ressonancia;
    as demais variaveis recebem uma expansao mais conservadora.
    """
    novo_config = copy.deepcopy(config)

    for nome, info in pressao.items():
        mn, mx, unid = config['geometric_vars'][nome]
        largura = mx - mn
        fator = (FATOR_EXPANSAO_PRINCIPAIS
                 if nome in VARS_PRIORITARIAS
                 else FATOR_EXPANSAO_SECUNDARIAS)
        delta = largura * fator

        if info['lado'] == 'inferior':
            novo_mn = max(mn - delta, 1e-3)
            novo_mx = mx
        else:
            novo_mn = mn
            novo_mx = mx + delta

        novo_config['geometric_vars'][nome] = (
            round(novo_mn, 4), round(novo_mx, 4), unid
        )

        log.info(
            f"  Expansao coordenada '{nome}' ({info['lado']}) | "
            f"top-{info['top_k']}: {info['n_borda']}/{info['top_k']} "
            f"({100.0 * info['fracao']:.0f}%) na borda | "
            f"[{mn}, {mx}] -> [{novo_mn:.4f}, {novo_mx:.4f}]"
        )

    return novo_config


def _salvar_historico_bounds(historico: list, base_dir: str) -> None:
    """Salva bounds e o motivo da decisao de remodelagem para auditoria."""
    linhas = []
    for rodada, config_i, pressao_i in historico:
        for nome, (mn, mx, unid) in config_i['geometric_vars'].items():
            info = pressao_i.get(nome, {})
            linhas.append({
                'rodada_bo': rodada,
                'variavel': nome,
                'min': mn,
                'max': mx,
                'unidade': unid,
                'pressao_borda': bool(info),
                'lado_pressao': info.get('lado', ''),
                'fracao_top_k_na_borda': info.get('fracao', np.nan),
                'n_top_k_na_borda': info.get('n_borda', np.nan),
                'top_k': info.get('top_k', np.nan),
                'posicao_melhor': info.get('pos_melhor', np.nan),
            })

    pd.DataFrame(linhas).to_csv(
        os.path.join(base_dir, 'historico_remodelagem_bounds.csv'), index=False
    )


def run_bayesian_optimization_adaptativo(config: dict, f_res_alvo: float, er_alvo: float,
                                          max_remodelagens: int = MAX_REMODELAGENS,
                                          margem_borda: float = MARGEM_BORDA,
                                          top_k_persistencia: int = TOP_K_PERSISTENCIA,
                                          min_fracao_borda: float = MIN_FRACAO_BORDA_PERSISTENTE,
                                          **kwargs) -> pd.DataFrame:
    """Executa BO com remodelagem robusta e limitada por par (f_res, Er).

    Regras:
      1. Uma unica BO e sempre feita com os bounds originais.
      2. Remodelagem so ocorre quando o melhor ponto esta na borda E os
         ``top_k_persistencia`` melhores pontos validos mostram pressao persistente
         na mesma borda.
      3. Todas as variaveis que satisfazem o criterio sao expandidas no MESMO
         passo; nao existe ajuste sequencial de uma variavel por rodada.
      4. A expansao nao recentra o intervalo no melhor ponto; ela apenas abre
         espaco na direcao da borda pressionada.
      5. No maximo ``max_remodelagens`` expansoes sao permitidas. Com o padrao 2,
         cada par recebe no maximo 3 BOs completas.
    """
    if max_remodelagens < 0:
        raise ValueError("max_remodelagens deve ser >= 0.")
    if top_k_persistencia < 2:
        raise ValueError("top_k_persistencia deve ser >= 2.")
    if not (0.5 <= min_fracao_borda <= 1.0):
        raise ValueError("min_fracao_borda deve estar entre 0.5 e 1.0.")

    config_base = copy.deepcopy(config)
    config_atual = copy.deepcopy(config_base)
    base_dir_original = config_base['base_dir']
    historico = []

    total_bo = max_remodelagens + 1
    df_result = None

    for rodada_bo in range(total_bo):
        if rodada_bo == 0:
            config_atual = copy.deepcopy(config_base)
        else:
            config_atual['base_dir'] = (
                f"{base_dir_original}_remodelagem{rodada_bo}"
            )

        log.info("\n" + "-" * 60)
        log.info(
            f"[ADAPTATIVO] BO {rodada_bo + 1}/{total_bo} | "
            f"f_res_alvo={f_res_alvo} | Er={er_alvo}"
        )
        log.info(f"[ADAPTATIVO] Bounds atuais: {config_atual['geometric_vars']}")
        log.info("-" * 60)

        df_result = run_bayesian_optimization(
            config_atual, f_res_alvo, er_alvo, **kwargs
        )

        pressao = _detectar_pressao_na_borda(
            df_result,
            config_atual,
            margem=margem_borda,
            top_k=top_k_persistencia,
            min_fracao=min_fracao_borda,
        )
        historico.append((rodada_bo + 1, copy.deepcopy(config_atual), pressao))

        if not pressao:
            log.info(
                "[ADAPTATIVO] Nenhuma pressao persistente na borda. "
                f"Par encerrado apos {rodada_bo + 1} BO(s)."
            )
            break

        if rodada_bo == total_bo - 1:
            log.warning(
                "[ADAPTATIVO] Limite de BOs atingido ({total_bo}). "
                f"Pressao residual: {list(pressao.keys())}. "
                "Par encerrado sem nova remodelagem."
            )
            break

        log.info(
            "[ADAPTATIVO] Pressao persistente confirmada em: "
            + ", ".join(
                f"{nome}({info['lado']}, {info['n_borda']}/{info['top_k']})"
                for nome, info in pressao.items()
            )
        )
        log.info(
            f"[ADAPTATIVO] Aplicando UMA expansao coordenada antes da BO "
            f"{rodada_bo + 2}/{total_bo}."
        )

        config_atual = _remodelar_bounds_coordenado(config_atual, pressao)

    _salvar_historico_bounds(historico, base_dir_original)

    # Resumo legivel do comportamento adaptativo.
    resumo_adaptativo = []
    for rodada, config_i, pressao_i in historico:
        resumo_adaptativo.append({
            'rodada_bo': rodada,
            'n_variaveis_pressao': len(pressao_i),
            'variaveis_pressao': ';'.join(pressao_i.keys()),
            'pressao_detalhes': ';'.join(
                f"{nome}:{info['lado']}:{info['n_borda']}/{info['top_k']}"
                for nome, info in pressao_i.items()
            )
        })
    pd.DataFrame(resumo_adaptativo).to_csv(
        os.path.join(base_dir_original, 'historico_adaptativo.csv'), index=False
    )

    return df_result

# =============================================================================
# LACO PRINCIPAL — dirigido por (f_res_alvo, er_alvo)
# =============================================================================
def run_bayesian_optimization(config: dict, f_res_alvo: float, er_alvo: float,
                               n_init: int | None = None, n_iter: int = N_ITER_BO,
                               freq_norm: float | None = None, s11_norm: float = 20.0,
                               casas_fabricacao: int = CASAS_FABRICACAO) -> pd.DataFrame:
    global log
    _validar_config(config)

    os.makedirs(config['base_dir'], exist_ok=True)
    log = configurar_log(os.path.join(config['base_dir'], 'bo_log.log'))

    nomes, bounds_arr = montar_espaco_busca(config)
    d = len(nomes)
    if n_init is None:
        n_init = max(N_INIT, 4 * (d + 1))

    freq_norm = freq_norm if freq_norm is not None else \
        (config['freq_sweep'][1] - config['freq_sweep'][0]) / 80

    log.info("=" * 60)
    log.info(f"BO | variaveis={nomes} | f_res_alvo={f_res_alvo} {config['freq_unit']} | er_alvo={er_alvo}")
    log.info(f"Orcamento: {n_init} iniciais + ate {n_iter} iteracoes | freq_norm={freq_norm:.5f}")
    log.info("=" * 60)
    _diagnosticar_orcamento(nomes, n_init, n_iter)

    hfss = None
    sweep_nome = None
    oModule = None
    hfss = conectar_hfss(config['project_path'])
    sweep_nome = configurar_hfss_simulacao(hfss, config)
    oModule = hfss.odesign.GetModule("ReportSetup")

    # Er e fixado uma unica vez — nao entra no espaco de busca nem no laco.
    hfss.materials[config['material_name']].permittivity = float(er_alvo)

    resultados_path = os.path.join(config['base_dir'], 'resultados_bo.csv')
    registros, X_obs, y_obs = [], [], []
    X_validos, y_validos, viavel_obs = [], [], []
    contador = [0]

    def _avaliar(x_vetor, fase, ei=None):
        contador[0] += 1
        valores = dict(zip(nomes, x_vetor))

        try:
            r = simular_ponto(hfss, oModule, sweep_nome, config, valores)
        except Exception as e:
            log.error(f"Erro inesperado ao simular {valores}: {e}", exc_info=True)
            r = None

        X_obs.append(list(x_vetor))
        viavel_obs.append(0 if r is None else 1)

        if r is None:
            log.info(f"  [REJEITADO] {valores} (fase={fase})")
            y_obs.append(PENALIDADE)
            registros.append({**valores, 'custo': PENALIDADE, 'fase': fase,
                               'iteracao': contador[0], 'ei': ei, 'rejeitado': True})
        else:
            custo = calcular_custo(r['f_res'], r['s11_min'], r['re_z11'],
                                    r['im_z11'], r['vswr'], f_res_alvo, freq_norm, s11_norm=s11_norm)
            y_obs.append(custo)
            X_validos.append(list(x_vetor))
            y_validos.append(custo)
            registros.append({**valores, **r, 'custo': custo, 'fase': fase,
                               'iteracao': contador[0], 'ei': ei, 'rejeitado': False})
            log.info(f"  {valores} -> f_res={r['f_res']} | S11={r['s11_min']} dB | custo={custo:.4f}")

        pd.DataFrame(registros).to_csv(resultados_path, index=False)

    try:
        for i, x in enumerate(amostragem_inicial(n_init, bounds_arr)):
            log.info(f"[INIT {i + 1}/{n_init}]")
            _avaliar(x, 'init')

        if len(X_validos) < 3:
            raise RuntimeError(
                "Poucos pontos validos na fase inicial (< 3) — revise "
                "geometric_vars, er_alvo ou os thresholds de S11.")

        xi_inicial, xi_final = 0.02, 0.002
        for it in range(n_iter):
            xi_it = xi_inicial + (xi_final - xi_inicial) * (it / max(n_iter - 1, 1))

            scaler_X = StandardScaler().fit(X_obs)
            X_obs_sc = scaler_X.transform(X_obs)

            usar_so_validos = len(X_validos) >= max(5, d + 1)
            if usar_so_validos:
                X_gp_sc, y_gp = scaler_X.transform(X_validos), np.array(y_validos)
            else:
                X_gp_sc, y_gp = X_obs_sc, np.array(y_obs)

            kernel = (ConstantKernel(1.0) *
                      Matern(length_scale=np.ones(d), length_scale_bounds=(1e-2, 1e2), nu=2.5)
                      + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-5, 1e0)))
            gp_custo = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                                 n_restarts_optimizer=5, random_state=SEMENTE)
            gp_custo.fit(X_gp_sc, y_gp)

            clf_viab = _treinar_classificador_viabilidade(X_obs_sc, viavel_obs)

            y_best = y_gp.min()
            x_next, aq_max = proximo_ponto(gp_custo, y_best, scaler_X, bounds_arr,
                                            seed=SEMENTE + it, clf_viab=clf_viab, xi=xi_it)

            log.info(f"[BO {it + 1}/{n_iter}] proximo={dict(zip(nomes, x_next))} | "
                     f"aquisicao={aq_max:.3e} | xi={xi_it:.4f} | "
                     f"GP com {len(y_gp)} pontos ({'validos' if usar_so_validos else 'todos'}) | "
                     f"classificador: {'ativo' if clf_viab is not None else 'inativo'}")

            if aq_max < EI_MIN:
                log.info("Aquisicao desprezivel - convergiu antes do orcamento total.")
                break

            _avaliar(x_next, 'bo', ei=aq_max)
            melhor_ate_agora = min(y_validos) if y_validos else float('nan')
            log.info(f"  melhor custo ate agora: {melhor_ate_agora:.4f}")

        df_result = pd.DataFrame(registros).sort_values('custo').reset_index(drop=True)
        df_result.to_csv(resultados_path, index=False)

        log.info("\n" + "=" * 60)
        log.info(f"MELHOR PONTO BRUTO ({len(df_result)} avaliacoes / "
                 f"{int((~df_result['rejeitado']).sum())} validos):")
        log.info(df_result.iloc[0].to_string())

        valores_raw = {nome: float(df_result.iloc[0][nome]) for nome in nomes}
        valores_fab = arredondar_fabricacao(valores_raw, casas=casas_fabricacao)
        custo_raw = float(df_result.iloc[0]['custo'])

        log.info(f"\nArredondando para fabricacao: {valores_raw} -> {valores_fab}")

        resultado_fab = simular_ponto(hfss, oModule, sweep_nome, config, valores_fab,
                                       retornar_curva=True, csv_prefixo='design_final')

        if resultado_fab is None:
            log.warning("Ponto arredondado REJEITADO na revalidacao — usando ponto bruto.")
            valores_fab = valores_raw
            resultado_fab = simular_ponto(hfss, oModule, sweep_nome, config, valores_fab,
                                           retornar_curva=True, csv_prefixo='design_final')

        df_curva_s11 = resultado_fab.pop('_curva_s11')
        custo_fab = calcular_custo(resultado_fab['f_res'], resultado_fab['s11_min'],
                                    resultado_fab['re_z11'], resultado_fab['im_z11'],
                                    resultado_fab['vswr'], f_res_alvo, freq_norm, s11_norm=s11_norm)

        log.info(f"Custo bruto: {custo_raw:.5f} | Custo pos-arredondamento: {custo_fab:.5f} "
                 f"(delta = {abs(custo_fab - custo_raw):.5f})")

        caminhos_fig = gerar_figuras_artigo(df_result, df_curva_s11, nomes, bounds_arr,
                                            config, f_res_alvo, resultado_fab, er_alvo)
        tabela = tabela_resultado_final(valores_fab, resultado_fab, config, er_alvo)
        tabela_path = os.path.join(config['base_dir'], 'tabela_resultado_final.csv')
        tabela.to_csv(tabela_path, index=False)

        imprimir_resumo_final(tabela, caminhos_fig, resultados_path, tabela_path)

    finally:
        # Fecha projetos + Desktop ao final de CADA rodada.
        encerrar_hfss(hfss)

    return df_result


# =============================================================================
# EXECUCAO EM GRADE — (frequencia, Er)
# =============================================================================
def _rotulo_grade(valor: float) -> str:
    texto = f"{float(valor):g}"
    return texto.replace('-', 'm').replace('.', 'p')


def run_grade_bayesiana(config: dict, frequencias: list[float], er_valores: list[float],
                        **kwargs) -> pd.DataFrame:
    """Executa todas as combinacoes frequencia x Er em sessoes HFSS independentes."""
    if not frequencias or not er_valores:
        raise ValueError("As listas de frequencias e Er da grade nao podem ser vazias.")

    frequencias = [float(v) for v in frequencias]
    er_valores = [float(v) for v in er_valores]
    grade_base_dir = os.path.join(config['base_dir'], 'grade_er_frequencia')
    os.makedirs(grade_base_dir, exist_ok=True)

    pares = [(f, er) for f in frequencias for er in er_valores]
    resumo = []
    log.info("\n" + "=" * 60)
    log.info(f"GRADE | {len(pares)} combinacoes: {len(frequencias)} frequencias x {len(er_valores)} valores de Er")
    log.info(f"Frequencias: {frequencias} {config['freq_unit']} | Er: {er_valores}")
    log.info("=" * 60)

    for indice, (f_res_alvo, er_alvo) in enumerate(pares, start=1):
        config_par = copy.deepcopy(config)
        pasta_par = os.path.join(grade_base_dir,
            f"f_{_rotulo_grade(f_res_alvo)}_{config['freq_unit']}_er_{_rotulo_grade(er_alvo)}")
        config_par['base_dir'] = pasta_par

        log.info("\n" + "#" * 60)
        log.info(f"GRADE {indice}/{len(pares)} | f_res_alvo={f_res_alvo} {config['freq_unit']} | er_alvo={er_alvo}")
        log.info(f"Diretorio: {pasta_par}")
        log.info("#" * 60)

        try:
            df_resultado = run_bayesian_optimization_adaptativo(
                config_par, f_res_alvo=f_res_alvo, er_alvo=er_alvo, **kwargs)
            melhor = df_resultado.iloc[0]
            resumo.append({
                'f_res_alvo': f_res_alvo, 'er_alvo': er_alvo, 'status': 'OK',
                'custo': float(melhor['custo']),
                'f_res_melhor': float(melhor['f_res']) if 'f_res' in melhor.index else np.nan,
                's11_min_melhor': float(melhor['s11_min']) if 's11_min' in melhor.index else np.nan,
                'diretorio': pasta_par, 'erro': ''})
            log.info(f"[GRADE] Concluido: f={f_res_alvo}, Er={er_alvo}")
        except Exception as e:
            log.error(f"[GRADE] Falha em f={f_res_alvo}, Er={er_alvo}: {e}", exc_info=True)
            resumo.append({
                'f_res_alvo': f_res_alvo, 'er_alvo': er_alvo, 'status': 'ERRO',
                'custo': np.nan, 'f_res_melhor': np.nan, 's11_min_melhor': np.nan,
                'diretorio': pasta_par, 'erro': repr(e)})

        # Folga pequena para o Windows/HFSS liberar recursos COM antes da proxima sessao.
        time.sleep(2)

    df_grade = pd.DataFrame(resumo)
    resumo_path = os.path.join(grade_base_dir, 'resumo_grade_er_frequencia.csv')
    df_grade.to_csv(resumo_path, index=False)
    log.info("\n" + "=" * 60)
    log.info(f"GRADE FINALIZADA | resumo: {resumo_path}")
    log.info("=" * 60)
    return df_grade


if __name__ == '__main__':
    df_grade = run_grade_bayesiana(
        ANTENNA_CONFIG,
        frequencias=GRADE_FREQUENCIAS,
        er_valores=GRADE_ER
    )