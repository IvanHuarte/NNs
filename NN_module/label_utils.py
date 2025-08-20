import ast

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table

def get_write_folder_from_model(config):

    name = config['model_NN']['selection']
    model_setup = config['model_NN'][name]
    model_label= config['CM']['selection'] +'_'+ name

    conditions = []
    labels = []
    if  name == "SplitTraining_ViT_MLP": 
        
        conditions = [model_setup['symm_2D_module'], model_setup['symm_2D_phase'], 
                      model_setup['symm_Z2_module'], model_setup['symm_Z2_phase']]
        labels = ["_2DM","_2DP","_Z2M","_Z2P"]
  

    elif  name == "SplitTraining_ViT_CNN":
        conditions = [model_setup['symm_2D_module'], model_setup['symm_Z2_module']]
        labels = ["_2DM","_Z2M"]

    symm=''
    for cond, label in zip(conditions,labels):
        if cond:
            symm+=label
            if label == '_Z2M':  symm += "t" if model_setup['trivial_Z2_module'] else "nt"
            if label == '_Z2P':  symm += "t" if model_setup['trivial_Z2_phase'] else "nt"
        
    return config['write_folder_sim'] + model_label+ symm + "/"

def display_simulation_settings(settings, n_cols=5):
    console=Console()
    cm_sel = settings["CM"]["selection"]
    nn_sel = settings["model_NN"]["selection"]

    cm_color= "red"
    nn_color= "bright_yellow"
    
    # Título principal
    title_text = f"[bold blue]🔧 CM:[/] [{cm_color}]{cm_sel}[/]   [bold blue]🧠 NN_architecture:[/] [{nn_color}]{nn_sel}[/]"
    console.rule(title_text)

    def _group_params(params_dict):
        # Filtra claves no relevantes
        items = [(k, v) for k, v in params_dict.items() if not k.endswith("_list")]
        grouped = [items[i:i + n_cols] for i in range(0, len(items), n_cols)]

        table = Table(show_header=False, box=None, pad_edge=False)
        for i in range(n_cols):
            table.add_column(justify="left")
        for group in grouped:
            row = [f"[bold]{k}[/]= {v}" for k, v in group]
            while len(row) < n_cols:
                row.append("")
            table.add_row(*row)
        return table
    
    # Size
    size = settings.get("size", None)
    if isinstance(size, (list, tuple)) and all(isinstance(x, int) for x in size):
        size_str = "x".join(map(str, size))
        console.print(f"[bold yellow]🧱 Size:[/] [cyan]{size_str}[/]\n")

    # Panel de acoplamiento
    cm_dict = settings["CM"].get(cm_sel, {})
    if isinstance(cm_dict, dict):
        table_cm = _group_params(cm_dict)
        panel_cm = Panel(table_cm, title=f"[bold]{cm_sel}[/]", border_style=cm_color)
        console.print(panel_cm)

    # Panel de red neuronal
    nn_dict = settings["model_NN"].get(nn_sel, {})
    if isinstance(nn_dict, dict):
        table_nn = _group_params(nn_dict)
        panel_nn = Panel(table_nn, title=f"[bold]{nn_sel}[/]", border_style=nn_color)
        console.print(panel_nn)

    console.rule("[bold green]")

def architecture_label(name, model_setup):

    if name == 'MLP':
        set_dim=[f"D({dim})" for dim in model_setup['hidden_alpha']] 
        set_act=[f"A({act})" for act in model_setup['activation']]
        setup='||'
        for i in range(len(model_setup['hidden_alpha'])):
            setup+=f" {set_dim[i]} |"
            setup+=f" {set_act[i]} |" if '0' not in set_act[i] else ''
        setup+="|"
        architecture = setup
        
    elif name == 'ViT':
        architecture = f"|| b: {model_setup['token_size']}  D_emb: {model_setup['embedding_d']}  heads: {model_setup['n_heads']} ||\n"
        architecture += f"|| n_blocks: {model_setup['n_blocks']}   ffn_layers: {model_setup['n_ffn_layers']} ||\n"
         
    elif name == 'CvT':
        architecture = f"|| n_CTE_ch: {model_setup['CTemb_channels']}  n_CPB: {model_setup['n_CP_blocks']}   CP_ch: {model_setup['CP_channels']} ||\n"
        architecture += f"|| heads: {model_setup['attn_heads']}  kernel: {model_setup['kernel']}   final_arch: {model_setup['final_architecture']} ||\n"
         
    elif name == 'SplitTraining_ViT_MLP':
        architecture = f"ViT \n"
        architecture += f"|| b: {model_setup['token_size']}  D_emb: {model_setup['embedding_d']}  heads: {model_setup['n_heads']} ||\n"
        architecture += f"|| n_blocks: {model_setup['n_blocks']}   ffn_layers: {model_setup['n_ffn_layers']} ||\n"
        architecture += f"MLP \n"
        set_dim=[f"D({dim})" for dim in model_setup['hidden_alpha']] 
        set_act=[f"A({act})" for act in model_setup['activation']]
        setup='||'
        for i in range(len(model_setup['hidden_alpha'])):
            setup+=f" {set_dim[i]} |"
            setup+=f" {set_act[i]} |" if '0' not in set_act[i] else ''
        setup+="|"
        architecture += setup
         
    elif name == 'SplitTraining_ViT_CNN':
        architecture = f"ViT \n"
        architecture += f"|| b: {model_setup['token_size']}  D_emb: {model_setup['embedding_d']}  heads: {model_setup['n_heads']} ||\n"
        architecture += f"|| n_blocks: {model_setup['n_blocks']}   ffn_layers: {model_setup['n_ffn_layers']} ||\n"
        architecture += f"CNN\n"
        architecture += f"|| block_channels: {model_setup['block_channels']}    kernel:{model_setup['kernel_size']}    n_ffn_lay:{model_setup['n_ffn_layers_cnn']} ||"
         
    elif name == 'SplitTraining_CvT_CNN':
        architecture = f"CvT \n"
        architecture += f"|| b: {model_setup['lattice_size']}  D_emb: {model_setup['CTemb_channels']}  heads: {model_setup['attn_heads']} ||\n"
        architecture += f"|| n_blocks: {model_setup['n_CP_blocks']}   CP_channels: {model_setup['CP_channels']} ||\n"
        architecture += f"|| kernel: {model_setup['kernel']}   final_architecture: {model_setup['final_architecture']} ||\n"
        architecture += f"CNN\n"
        architecture += f"|| block_channels: {model_setup['block_channels_cnn']}    kernel:{model_setup['kernel_size_cnn']}    n_ffn_lay:{model_setup['n_ffn_layers_cnn']} ||"
    
    elif name == 'SplitTraining_CvT_CvT':
        architecture = f"CvT 1\n"
        architecture += f"|| n_CTE_ch: {model_setup['CTemb_channels_1']}  n_CPB: {model_setup['n_CP_blocks_1']}   CP_ch: {model_setup['CP_channels_1']} ||\n"
        architecture += f"|| heads: {model_setup['attn_heads_1']}  kernel: {model_setup['kernel_1']}   final_arch: {model_setup['final_architecture_1']} ||\n"
        architecture += f"CvT 2\n"
        architecture += f"|| n_CTE_ch: {model_setup['CTemb_channels_2']}  n_CPB: {model_setup['n_CP_blocks_2']}   CP_ch: {model_setup['CP_channels_2']} ||\n"
        architecture += f"|| heads: {model_setup['attn_heads_2']}  kernel: {model_setup['kernel_2']}   final_arch: {model_setup['final_architecture_2']} ||\n"
         
    return architecture
    
def get_filenames_from_settings(cm_name, nn_name, **kwargs):
    
    model_label = cm_name + '_' + nn_name
    size=kwargs['size']

    if cm_name == 'Oxalate':
        strength = kwargs['strength'] ; theta = kwargs['theta'] ; phi = kwargs['phi']
        cparams= f"_strength_{strength}_theta_{theta}_phi_{phi}"
        call_params = r"$a = %.1f$  $\theta = %.1f$  $\phi = %.1f$"%(strength,theta,phi)

    elif cm_name == 'Chain_YYZZ':
        fields = kwargs['fields'] ; couplings = kwargs['couplings']
        flat_fields=''
        field_values=''
        call_params=''
        for f,v in zip(['X','Y','Z'], fields):
            flat_fields += f + '_'
            field_values += f"{v}" + '_'
            call_params += f"{f}:{v}  "

        flat_couplings=''
        coupling_values=''
        for f,v in zip(['XX','YY','ZZ'], couplings):
            flat_couplings += f + '_'
            coupling_values += f"{v}" + '_'
            call_params += f"{f}:{v}  "

    if nn_name == 'MLP':
        alphas = kwargs['hidden_alpha'] ; activation = kwargs['activation'] 
        act_label=''
        dim_label=''
        for act in activation:
            act_label += f"{act}_"
        for a in alphas:
            dim_label += f"{a}_"
        nnparams = f"_alphas_{dim_label}_activations_{act_label}"

    elif nn_name == 'ViT':
        token_size = kwargs['token_size'] ; embedding_d = kwargs['embedding_d'] ; n_heads = kwargs['n_heads'] 
        n_blocks = kwargs['n_blocks'] ; n_ffn_layers = kwargs['n_ffn_layers']
        nnparams = f"_b_{token_size[0]}x{token_size[1]}_Demb_{embedding_d}_heads_{n_heads}_blocks_{n_blocks}_ffn_lay_{n_ffn_layers}"
    
    elif nn_name == 'CvT':
        n_CP_blocks = kwargs['n_CP_blocks'] ;  CTemb_channels = kwargs['CTemb_channels'] 
        CP_channels = kwargs['CP_channels'] ; attn_heads = kwargs['attn_heads']
        kernel = kwargs['kernel'] ; final_architecture = ast.literal_eval(kwargs['final_architecture'] )
        kernel_label = blocks_label = emb_ch_label = cp_ch_label = heads_label = arch_label = ''
        for i in range(len(n_CP_blocks)):
            blocks_label += f"{n_CP_blocks[i]}_"
            emb_ch_label += f"{CTemb_channels[i]}_"
            cp_ch_label += f"{CP_channels[i]}_"
            heads_label += f"{attn_heads[i]}_"        
        for i in range(len(final_architecture)):
            arch_label += f"{final_architecture[i]}_"
        kernel_label += f"{kernel[0]}x{kernel[1]}_"
        nnparams = f"_blocks_{blocks_label}emb_ch_{emb_ch_label}cp_ch_{cp_ch_label}heads_{heads_label}kernel_{kernel_label}finarch_{arch_label}"

    elif nn_name == 'SplitTraining_ViT_MLP':
        token_size = kwargs['token_size'] ; embedding_d = kwargs['embedding_d'] ; n_heads = kwargs['n_heads']   # ViT
        n_blocks = kwargs['n_blocks'] ; n_ffn_layers = kwargs['n_ffn_layers']
        alphas = kwargs['hidden_alpha'] ; activation = kwargs['activation']     # MLP
        act_label=''
        dim_label=''
        for act in activation:
            act_label += f"{act}_"
        for a in alphas:
            dim_label += f"{a}_"
        
        nnparams= f"_ViT_b_{token_size[0]}x{token_size[1]}_Demb_{embedding_d}_heads_{n_heads}_blocks_{n_blocks}_ffn_lay_{n_ffn_layers}__MLP_alphas_{dim_label}_activations_{act_label}"
    
    elif nn_name == 'SplitTraining_ViT_CNN':
        token_size = kwargs['token_size'] ; embedding_d = kwargs['embedding_d'] ; n_heads = kwargs['n_heads']   # ViT
        n_blocks = kwargs['n_blocks'] ; n_ffn_layers = kwargs['n_ffn_layers']
        block_channels = kwargs['block_channels'] ; kernel_size = kwargs['kernel_size'] ; n_ffn_layers_cnn = kwargs['n_ffn_layers_cnn']     # CNN
        cha_label=''
        for ch in block_channels:
            cha_label += f"{ch}_"
        nnparams = f"_channels_{cha_label}kernel_{kernel_size[0]}x{kernel_size[1]}_n_ffn_lay_{n_ffn_layers_cnn}"

    elif nn_name == 'SplitTraining_CvT_CNN':
        # CvT params
        n_CP_blocks = kwargs['n_CP_blocks'] ;  CTemb_channels = kwargs['CTemb_channels'] 
        CP_channels = kwargs['CP_channels'] ; attn_heads = kwargs['attn_heads']
        kernel = kwargs['kernel'] ; final_architecture = ast.literal_eval(kwargs['final_architecture'] )

        kernel_label = blocks_label = emb_ch_label = cp_ch_label = heads_label = arch_label = ''
        for i in range(len(n_CP_blocks)):
            blocks_label += f"{n_CP_blocks[i]}_"
            emb_ch_label += f"{CTemb_channels[i]}_"
            cp_ch_label += f"{CP_channels[i]}_"
            heads_label += f"{attn_heads[i]}_" 
        for i in range(len(final_architecture)):       
            arch_label += f"{final_architecture[i]}_"
        kernel_label += f"{kernel[0]}x{kernel[1]}_"
        # CNN params
        block_channels_cnn = kwargs['block_channels_cnn'] ; kernel_size_cnn = kwargs['kernel_size_cnn'] ; n_ffn_layers_cnn = kwargs['n_ffn_layers_cnn']     # CNN
        cha_label_cnn=''
        for ch in block_channels_cnn:
            cha_label_cnn += f"{ch}_"
        nnparams = f"CvT_blocks_{blocks_label}emb_ch_{emb_ch_label}cp_ch_{cp_ch_label}heads_{heads_label}kernel_{kernel_label}finarch_{arch_label}"
        nnparams += f"_CNN_channels_{cha_label_cnn}kernel_{kernel_size_cnn[0]}x{kernel_size_cnn[0]}_n_ffn_lay_{n_ffn_layers_cnn}"
    
    elif nn_name == 'SplitTraining_CvT_CvT':
        # CvT 1 params
        n_CP_blocks_1 = kwargs['n_CP_blocks_1'] ;  CTemb_channels_1 = kwargs['CTemb_channels_1'] 
        CP_channels_1 = kwargs['CP_channels_1'] ; attn_heads_1 = kwargs['attn_heads_1']
        kernel_1 = kwargs['kernel_1'] ; final_architecture_1 = ast.literal_eval(kwargs['final_architecture_1'] )

        kernel_label_1 = blocks_label_1 = emb_ch_label_1 = cp_ch_label_1 = heads_label_1 = arch_label_1 = ''
        for i in range(len(n_CP_blocks_1)):
            blocks_label_1 += f"{n_CP_blocks_1[i]}_"
            emb_ch_label_1 += f"{CTemb_channels_1[i]}_"
            cp_ch_label_1 += f"{CP_channels_1[i]}_"
            heads_label_1 += f"{attn_heads_1[i]}_" 
        for i in range(len(final_architecture_1)):       
            arch_label_1 += f"{final_architecture_1[i]}_"
        kernel_label_1 += f"{kernel_1[0]}x{kernel_1[0]}_" 

        # CvT 2 params
        n_CP_blocks_2 = kwargs['n_CP_blocks_2'] ;  CTemb_channels_2 = kwargs['CTemb_channels_2'] 
        CP_channels_2 = kwargs['CP_channels_2'] ; attn_heads_2 = kwargs['attn_heads_2']
        kernel_2 = kwargs['kernel_2'] ; final_architecture_2 = ast.literal_eval(kwargs['final_architecture_2'] ) 

        kernel_label_2 = blocks_label_2 = emb_ch_label_2 = cp_ch_label_2 = heads_label_2 = arch_label_2 = ''
        for i in range(len(n_CP_blocks_2)):
            blocks_label_2 += f"{n_CP_blocks_2[i]}_" 
            emb_ch_label_2 += f"{CTemb_channels_2[i]}_" 
            cp_ch_label_2 += f"{CP_channels_2[i]}_" 
            heads_label_2 += f"{attn_heads_2[i]}_"        
        for i in range(len(final_architecture_2)):
            arch_label_2 += f"{final_architecture_2[i]}_" 
        kernel_label_2 += f"{kernel_2[0]}x{kernel_2[0]}_" 

        nnparams= f"CvT1_blocks_{blocks_label_1}emb_ch_{emb_ch_label_1}cp_ch_{cp_ch_label_1}heads_{heads_label_1}kernel_{kernel_label_1}finarch_{arch_label_1}"
        nnparams += f"_CvT2_blocks_{blocks_label_2}emb_ch_{emb_ch_label_2}cp_ch_{cp_ch_label_2}heads_{heads_label_2}kernel_{kernel_label_2}finarch_{arch_label_2}"

    sim_label = model_label + f"_simulation_{size[0]}x{size[1]}"+ cparams + nnparams
    ED_label = model_label + f"_xED_{size[0]}x{size[1]}"+cparams
    json_label = model_label+f"_results_{size[0]}x{size[1]}"+ cparams + nnparams
    title_label_callback = f"Callback "+model_label+" " + call_params + f"  ({size[0]}x{size[1]})"

    return sim_label, ED_label, json_label, title_label_callback
