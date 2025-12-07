import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib
import scipy.stats as stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from statsmodels.miscmodels.ordinal_model import OrderedModel
import statsmodels.api as sm
import os
import re

# ==========================================
# 設定エリア
# ==========================================
DATA_FILE = "Data.csv"
INDEX_FILE = "index.csv"
OUTPUT_DIR = "./results/"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ==========================================
# 分析エンジン
# ==========================================
class SocialAnalyzer:
    def __init__(self, data_path, index_path):
        print("=== 初期化プロセス開始 ===")
        
        # 1. Indexファイルの読み込みと解析
        self.groups = {}      # { "グループ名": [変数名リスト] }
        self.rev_vars = []    # 逆転項目の変数名リスト
        self.load_index(index_path)
        
        # 2. データファイルの読み込み
        print(f"\nLoading data from {data_path}...")
        try:
            try:
                self.df = pd.read_csv(data_path, encoding='utf-8')
            except UnicodeDecodeError:
                self.df = pd.read_csv(data_path, encoding='shift-jis')
            print(f"読み込み完了: {len(self.df)} 行のデータがあります。")
        except FileNotFoundError:
            print(f"エラー: {data_path} が見つかりません。")
            return

        # 3. 逆転項目の処理 (上書き)
        self.process_reverse_items()
        
        # 4. 前処理済みデータの保存
        save_path = os.path.join(OUTPUT_DIR, "preprocessed_data.csv")
        self.df.to_csv(save_path, index=False, encoding='utf-8_sig')
        print(f"\n>> 前処理済みのデータを保存しました: {save_path}")
        print(">> 今後の分析にはこのデータが使用されます。")

    def load_index(self, index_path):
        """index.csv を読み込み、変数のグループ化と逆転項目の特定を行う"""
        print(f"Loading index from {index_path}...")
        try:
            idx_df = pd.read_csv(index_path)
            # カラム名が想定通りかチェック (code, variable)
            if 'code' not in idx_df.columns or 'variable' not in idx_df.columns:
                print("エラー: index.csv に 'code' または 'variable' カラムがありません。")
                return

            for _, row in idx_df.iterrows():
                code = str(row['code'])
                var_name = str(row['variable'])
                
                # --- グループ名の抽出ロジック ---
                # code (例: Relational_Mobility_4_rev) から _数字 や _rev を除去してグループ名にする
                # 正規表現: 末尾の (_rev) や (_数字) を繰り返し削除
                group_name = re.sub(r'(_\d+)?(_rev|_R)?$', '', code) # 末尾除去
                group_name = group_name.rstrip('_') # 末尾のアンダースコア除去
                
                # グループ辞書に追加
                if group_name not in self.groups:
                    self.groups[group_name] = []
                self.groups[group_name].append(var_name)
                
                # --- 逆転項目の判定 ---
                if '_rev' in code.lower():
                    self.rev_vars.append(var_name)

            print(f"Index解析完了: {len(self.groups)} 個の変数グループを定義しました。")
            
        except FileNotFoundError:
            print(f"エラー: {index_path} が見つかりません。")

    def process_reverse_items(self):
        """逆転項目を検出し、ユーザー入力に基づいて値を反転(上書き)する"""
        if not self.rev_vars:
            return

        # 実際にデータフレームに存在する逆転項目のみを対象にする
        valid_rev_vars = [v for v in self.rev_vars if v in self.df.columns]
        
        if not valid_rev_vars:
            return

        print(f"\n【逆転処理】index.csv に基づき {len(valid_rev_vars)} 個の逆転項目を検出しました。")
        print(f"例: {valid_rev_vars[:3]} ...")
        
        print("これらを反転させるために、尺度の範囲を教えてください。")
        print("データが書き換えられます (例: 1->7, 7->1)")
        
        try:
            min_val = float(input("尺度の最小値 (Min): "))
            max_val = float(input("尺度の最大値 (Max): "))
            adjustment_val = max_val + min_val
            
            for col in valid_rev_vars:
                # 値を反転して上書き
                self.df[col] = adjustment_val - self.df[col]
            
            print(f">> 完了: {len(valid_rev_vars)} 個の変数を反転させました。")
            
        except ValueError:
            print("数値が無効です。逆転処理をスキップします（分析結果が不正確になる可能性があります）。")

    def get_group_selection(self, message, multi=False):
        print(f"\n--- {message} ---")
        group_names = list(self.groups.keys())
        
        # グループ一覧を表示
        for i, g_name in enumerate(group_names):
            vars_in_group = self.groups[g_name]
            valid_vars = [v for v in vars_in_group if v in self.df.columns]
            print(f"[{i}] {g_name} (項目数: {len(valid_vars)})")
        
        # 個別変数の選択オプション
        print(f"[{len(group_names)}] (個別の変数を手動で選ぶ)")

        while True:
            try:
                user_input = input("番号を選択してください: ")
                idx = int(user_input)
                
                # グループ選択
                if 0 <= idx < len(group_names):
                    selected_group = group_names[idx]
                    selected_vars = [v for v in self.groups[selected_group] if v in self.df.columns]
                    for i in len(selected_vars):
                        print(f"[{i + 1}] : {selected_vars[i]}")
                    return selected_vars # 元々リストなのでそのまま
                
                # 個別選択モード
                elif idx == len(group_names):
                    val = self.get_column_selection_manual(multi)
                    if isinstance(val, str):
                        return [val]
                    return val
                
                else:
                    print("無効な番号です。")
            except ValueError:
                print("番号を入力してください。")

    def get_column_selection_manual(self, multi=False):
        """従来の手動カラム選択"""
        print("\n--- 個別変数選択モード ---")
        cols = list(self.df.columns)
        for i, col in enumerate(cols):
            print(f"[{i}] {col}")
        
        while True:
            try:
                inp = input("番号を入力 (カンマ区切り): ")
                indices = [int(x.strip()) for x in inp.split(',')]
                selected = [cols[i] for i in indices if 0 <= i < len(cols)]
                if not multi and len(selected) > 1:
                    print("1つだけ選んでください。")
                    continue
                return selected if multi else selected[0]
            except:
                pass

    # --------------------------------------
    # 分析メソッド (グループ選択UIを使用するように変更)
    # --------------------------------------
    
    def run_basic_stats(self):
        print("\n=== 基本統計量 ===")
        # 全変数の統計量を出すか、グループごとに絞るか
        # ここではシンプルに全変数の統計量を出して保存する
        numeric_df = self.df.select_dtypes(include=[np.number])
        stats = numeric_df.describe()
        print(stats)
        stats.to_csv(os.path.join(OUTPUT_DIR, "basic_stats.csv"))
        print("basic_stats.csv に保存しました。")
    
    def run_group_comparison(self):
        print("\n=== グループ間比較 (検定 + 効果量) ===")
        
        # 1. グループ変数（独立変数）の選択
        print(">> 群分けに使う変数を選んでください (例: treatment)")
        group_col_list = self.get_group_selection("グループ変数を選択")
        
        if len(group_col_list) != 1:
            print(f"エラー: グループ変数は1つだけ指定してください。(選択数: {len(group_col_list)})")
            return
        
        group_col = group_col_list[0]
        
        # 2. 従属変数（検定したい変数）の選択
        print("\n>> 比較・検定したい変数（従属変数）を選んでください")
        target_cols = self.get_group_selection("従属変数を選択", multi=True)
        
        if not target_cols:
            return

        # 3. 検定モードの選択
        print("\n>> 検定手法を選んでください")
        print("[1] パラメトリック検定 (t検定 / ANOVA) - 正規分布を仮定")
        print("[2] ノンパラメトリック検定 (Mann-Whitney U / Kruskal-Wallis) - 順序尺度や非正規分布")
        
        mode = input("番号を選択 (デフォルト: 1): ")
        is_non_parametric = (mode == '2')
        
        # 結果格納用リスト
        results = []

        print(f"\n分析実行中... (グループ変数: {group_col}, モード: {'Non-Parametric' if is_non_parametric else 'Parametric'})")
        
        for target in target_cols:
            # データ準備 (欠損除去)
            data = self.df[[group_col, target]].dropna()
            
            # グループごとのデータに分割
            groups = data.groupby(group_col)[target].apply(list)
            group_keys = list(groups.keys())
            group_values = list(groups)
            
            n_groups = len(group_keys)
            N = len(data) # 総サンプル数
            
            # 結果辞書の初期化
            res = {
                "Variable": target,
                "Test_Type": "-",
                "Statistic": 0.0,
                "p_value": 1.0,
                "Significance": "",
                "Effect_Size_Type": "-",
                "Effect_Size": 0.0,
                "Note": str(group_keys)
            }

            if n_groups < 2:
                res["Note"] = "エラー: グループ不足"
            
            elif n_groups == 2:
                # === 2群の比較 ===
                g1 = np.array(group_values[0])
                g2 = np.array(group_values[1])
                n1, n2 = len(g1), len(g2)

                if is_non_parametric:
                    # --- Mann-Whitney U検定 ---
                    res["Test_Type"] = "Mann-Whitney U"
                    stat_val, p_val = stats.mannwhitneyu(g1, g2, alternative='two-sided')
                    
                    # 効果量: Rank-biserial correlation (r)
                    # r = 1 - (2U) / (n1 * n2)
                    res["Effect_Size_Type"] = "r (Rank-biserial)"
                    res["Effect_Size"] = 1 - (2 * stat_val) / (n1 * n2)
                    
                else:
                    # --- Welch's t検定 ---
                    res["Test_Type"] = "Welch t-test"
                    stat_val, p_val = stats.ttest_ind(g1, g2, equal_var=False)
                    
                    # 効果量: Cohen's d
                    # d = (mean1 - mean2) / pooled_std
                    # pooled_std = sqrt( ((n1-1)s1^2 + (n2-1)s2^2) / (n1+n2-2) )
                    m1, m2 = np.mean(g1), np.mean(g2)
                    v1, v2 = np.var(g1, ddof=1), np.var(g2, ddof=1)
                    pooled_std = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
                    
                    res["Effect_Size_Type"] = "Cohen's d"
                    res["Effect_Size"] = (m1 - m2) / pooled_std
                
                res["Statistic"] = stat_val
                res["p_value"] = p_val
                
            else:
                # === 3群以上の比較 ===
                if is_non_parametric:
                    # --- Kruskal-Wallis検定 ---
                    res["Test_Type"] = "Kruskal-Wallis"
                    stat_val, p_val = stats.kruskal(*group_values)
                    
                    # 効果量: Eta-squared (H)
                    # eta^2 = (H - k + 1) / (N - k)
                    k = n_groups
                    res["Effect_Size_Type"] = "Eta-sq (H)"
                    if N - k != 0:
                        res["Effect_Size"] = (stat_val - k + 1) / (N - k)
                    else:
                        res["Effect_Size"] = 0 # エラー回避
                    
                else:
                    # --- 一元配置分散分析 (One-way ANOVA) ---
                    res["Test_Type"] = "One-way ANOVA"
                    stat_val, p_val = stats.f_oneway(*group_values)
                    
                    # 効果量: Eta-squared
                    # eta^2 = (F * df_between) / (F * df_between + df_within)
                    k = n_groups
                    df_bg = k - 1
                    df_wg = N - k
                    res["Effect_Size_Type"] = "Eta-squared"
                    res["Effect_Size"] = (stat_val * df_bg) / (stat_val * df_bg + df_wg)
                
                res["Statistic"] = stat_val
                res["p_value"] = p_val
            
            # 有意差の判定マーク
            if res["p_value"] < 0.01:
                res["Significance"] = "**"
            elif res["p_value"] < 0.05:
                res["Significance"] = "*"
            elif res["p_value"] < 0.1:
                res["Significance"] = "+"
                
            results.append(res)

        # 4. 結果の表示と保存
        if results:
            results_df = pd.DataFrame(results)
            # 見やすく表示
            print("\n【検定結果一覧】")
            # 表示カラムを指定
            cols = ["Variable", "Test_Type", "Statistic", "p_value", "Significance", "Effect_Size_Type", "Effect_Size"]
            print(results_df[cols])
            
            # CSV保存
            filename = "group_comparison_nonparam.csv" if is_non_parametric else "group_comparison_param.csv"
            save_path = os.path.join(OUTPUT_DIR, filename)
            results_df.to_csv(save_path, index=False, encoding='utf-8_sig')
            print(f"\n>> 結果を保存しました: {save_path}")
            print("   (Effect Size目安: d=0.2/0.5/0.8, r=0.1/0.3/0.5, eta2=0.01/0.06/0.14)")

    def run_ordinal_regression(self):
        print("\n=== 順序ロジスティック回帰分析 ===")
        print("注: リッカート尺度などの「順序変数」を従属変数(Y)とする場合に使用します。")
        
        # 1. 従属変数(Y)の選択
        print(">> 従属変数(Y)を選んでください（順序尺度である必要があります）。")
        y_col = self.get_column_selection_manual(multi=False)
        if isinstance(y_col, list): y_col = y_col[0] # リストで返ってきた場合の保険
        
        # 2. 独立変数(X)の選択
        print("\n>> 独立変数(X)を選んでください。")
        x_vars = []
        while True:
            selection = self.get_group_selection("追加する変数(群)を選んでください")
            x_vars.extend(selection)
                
            cont = input("さらに変数を追加しますか？ (y/n): ")
            if cont.lower() != 'y':
                break
        
        # 重複除去
        x_vars = list(set(x_vars))
        
        # YがXに含まれていたら削除
        if y_col in x_vars:
            x_vars.remove(y_col)
            
        print(f"\nモデル構築中: {y_col} ~ {x_vars}")
        
        # データ準備
        data_reg = self.df[[y_col] + x_vars].dropna()
        
        if len(data_reg) < 5:
            print("エラー: 有効なデータ件数が少なすぎます。")
            return

        try:
            # モデル定義 (distr='logit' でロジスティック分布を指定)
            # OrderedModelは Y, X の順で渡す
            model = OrderedModel(data_reg[y_col], data_reg[x_vars], distr='logit')
            
            # フィッティング (method='bfgs' は数値計算が比較的安定しやすい)
            res = model.fit(method='bfgs', disp=False)
            
            print("\n" + "="*40)
            print(res.summary())
            print("="*40)
            
            # 結果保存
            save_file = os.path.join(OUTPUT_DIR, "ordinal_regression_summary.txt")
            with open(save_file, "w") as f:
                f.write(res.summary().as_text())
            print(f">> 結果を保存しました: {save_file}")
            
            # オッズ比の表示（解釈しやすいため）
            print("\n【参考: オッズ比 (Odds Ratios)】")
            print("係数(Coef)の指数をとった値です。1より大きければ正の影響、1未満なら負の影響を表します。")
            params = res.params
            conf = res.conf_int()
            conf['Odds Ratio'] = params
            conf.columns = ['2.5%', '97.5%', 'Odds Ratio']
            # 指数変換
            out = np.exp(conf)
            print(out[['Odds Ratio', '2.5%', '97.5%']])

        except Exception as e:
            print(f"\n計算エラーが発生しました: {e}")
            print("考えられる原因:")
            print("- サンプルサイズに対して変数が多すぎる")
            print("- 従属変数のカテゴリごとの度数が0または極端に少ない")
            print("- データ間の相関が高すぎる (多重共線性)")

    def run_regression(self):
        print("\n=== 回帰分析 ===")
        # 従属変数は単一選択 (手動選択モード推奨)
        print(">> 従属変数(Y)を選びます。")
        y = self.get_column_selection_manual(multi=False) # Yはたいてい1つの変数なので手動選択へ
        
        print(">> 独立変数(X)を選びます。グループ選択も可能です。")
        # グループで選ぶか手動で選ぶか
        x_vars = []
        while True:
            selection = self.get_group_selection("追加する変数(群)を選んでください")
            if isinstance(selection, list):
                x_vars.extend(selection)
            else:
                x_vars.append(selection)
                
            cont = input("さらに変数を追加しますか？ (y/n): ")
            if cont.lower() != 'y':
                break
        
        # 重複除去
        x_vars = list(set(x_vars))
        print(f"独立変数リスト: {x_vars}")

        data_reg = self.df[[y] + x_vars].dropna()
        Y_val = data_reg[y]
        X_val = sm.add_constant(data_reg[x_vars])
        
        model = sm.OLS(Y_val, X_val).fit()
        print(model.summary())
        with open(os.path.join(OUTPUT_DIR, "regression.txt"), "w") as f:
            f.write(model.summary().as_text())

    def run_pca(self):
        print("\n=== 主成分分析 (PCA) ===")
        items = self.get_group_selection("分析に使用する変数群を選んでください")
        
        data_for_pca = self.df[items].dropna()
        if data_for_pca.empty:
            print("有効なデータがありません。")
            return

        scaler = StandardScaler()
        X_std = scaler.fit_transform(data_for_pca)
        
        pca = PCA()
        pca.fit(X_std)
        
        print("\n【寄与率】")
        for i, r in enumerate(pca.explained_variance_ratio_):
            print(f"PC{i+1}: {r:.3f}")
            
        loadings = pd.DataFrame(pca.components_.T, index=items, columns=[f"PC{i+1}" for i in range(len(items))])
        loadings.to_csv(os.path.join(OUTPUT_DIR, "pca_loadings.csv"))
        print("因子負荷量を保存しました。")

        # 得点追加
        scores = pca.transform(X_std)
        try:
            n = int(input("保存する主成分数 (例: 1): "))
        except:
            n = 0
        for i in range(n):
            col_name = f"PCA_{items[0][:5]}_PC{i+1}" # 変数名の一部を使って命名
            self.df.loc[data_for_pca.index, col_name] = scores[:, i]
            print(f"変数追加: {col_name}")

    def run_reliability(self):
        print("\n=== 信頼性分析 (Cronbach's alpha) ===")
        # グループ選択UIを呼び出す
        items = self.get_group_selection("分析したい尺度(グループ)を選んでください")
        
        if len(items) < 2:
            print("エラー: 項目数が足りません(2つ以上必要)。")
            return

        df_items = self.df[items].dropna()
        k = df_items.shape[1]
        
        # 分散計算
        sum_item_var = df_items.var(ddof=1).sum()
        total_score_var = df_items.sum(axis=1).var(ddof=1)
        
        if total_score_var == 0:
            alpha = 0.0
        else:
            alpha = (k / (k - 1)) * (1 - (sum_item_var / total_score_var))
        
        print("\n--------------------------------")
        print(f" 項目数 (k): {k}")
        print(f" 対象データ数 (N): {len(df_items)}")
        print(f" クロンバッハのα係数: {alpha:.4f}")
        print("--------------------------------")

        # 結果の保存
        with open(os.path.join(OUTPUT_DIR, "reliability_result.txt"), "w", encoding="utf-8") as f:
            f.write(f"Items: {items}\n")
            f.write(f"N: {len(df_items)}\n")
            f.write(f"Cronbach's Alpha: {alpha:.4f}\n")
        print("結果をテキストファイルに保存しました。")
        
        # 0.8以上なら高い、などの目安を表示しても親切です
        if alpha >= 0.8:
            print(">> 判定: 十分な信頼性があります (High consistency)")
        elif alpha >= 0.7:
            print(">> 判定: 許容できる信頼性です (Acceptable)")
        else:
            print(">> 判定: 信頼性は低めです。項目の再検討が必要かもしれません。")
        
        # 平均値変数の作成提案
        make_mean = input("この尺度の「平均値」を変数として追加しますか？ (y/n): ")
        if make_mean.lower() == 'y':
            # グループ名を取得するために逆引きするか、単純にNamingするか
            name = input("変数名を入力してください (例: Mean_RelationalMobility): ")
            self.df[name] = df_items.mean(axis=1)
            print(f"変数 '{name}' を追加しました。")

    def run_mediation_analysis(self):
        print("\n=== 媒介分析 (ブートストラップ法) ===")
        print("HayesのModel 4 (単純媒介) に相当します。")
        print("パス図: X (独立) --> M (媒介) --> Y (従属)")
        
        # 1. 変数選択
        print("\n>> 1. 従属変数 (Y) を選んでください")
        y_col = self.get_column_selection_manual(multi=False)
        if isinstance(y_col, list): y_col = y_col[0]

        print("\n>> 2. 独立変数 (X) を選んでください")
        x_col = self.get_column_selection_manual(multi=False)
        if isinstance(x_col, list): x_col = x_col[0]
        
        print("\n>> 3. 媒介変数 (M) を選んでください")
        m_col = self.get_column_selection_manual(multi=False)
        if isinstance(m_col, list): m_col = m_col[0]

        # 統制変数の選択（任意）
        print("\n>> 4. 統制変数 (Covariates) を追加しますか？")
        cov_vars = []
        if input("追加する場合は 'y' を入力: ").lower() == 'y':
            cov_vars = self.get_group_selection("統制変数を選んでください")
        
        # 重複チェック
        used_cols = {y_col, x_col, m_col}
        cov_vars = [c for c in cov_vars if c not in used_cols]
        
        all_cols = [y_col, x_col, m_col] + cov_vars
        
        # データ準備 (欠損値除去と数値変換)
        df_model = self.df[all_cols].apply(pd.to_numeric, errors='coerce').dropna()
        
        if len(df_model) < 10:
            print("エラー: 有効なサンプルサイズが少なすぎます。")
            return

        print(f"\n分析対象データ数: N = {len(df_model)}")
        print("ブートストラップ検定を実行中... (処理に数秒かかる場合があります)")
        
        # === 内部関数: 回帰係数を計算して間接効果を返す ===
        def calculate_effects(data_sample):
            # Model 1: M ~ X + Covs (Path a)
            X_m = sm.add_constant(data_sample[[x_col] + cov_vars])
            Y_m = data_sample[m_col]
            model_m = sm.OLS(Y_m, X_m).fit()
            a_path = model_m.params[x_col]
            
            # Model 2: Y ~ X + M + Covs (Path c' and b)
            X_y = sm.add_constant(data_sample[[x_col, m_col] + cov_vars])
            Y_y = data_sample[y_col]
            model_y = sm.OLS(Y_y, X_y).fit()
            b_path = model_y.params[m_col]
            c_prime_path = model_y.params[x_col] # 直接効果
            
            indirect_effect = a_path * b_path
            total_effect = c_prime_path + indirect_effect
            
            return a_path, b_path, c_prime_path, indirect_effect, total_effect

        # 1. 観測データでの推定
        a_obs, b_obs, c_prime_obs, ind_obs, tot_obs = calculate_effects(df_model)
        
        # 2. ブートストラップ法による信頼区間の算出
        n_boot = 2000 # ブートストラップ回数 (通常1000~5000)
        boot_indirect_effects = []
        
        try:
            for _ in range(n_boot):
                # リサンプリング (重複を許してN個取り出す)
                sample_df = resample(df_model, replace=True, n_samples=len(df_model))
                
                # 計算できないサンプル(分散0など)が出た場合スキップ
                try:
                    _, _, _, ind_boot, _ = calculate_effects(sample_df)
                    boot_indirect_effects.append(ind_boot)
                except:
                    continue
            
            if len(boot_indirect_effects) < 100:
                print("エラー: ブートストラップ計算に失敗しました。")
                return

            # 95%信頼区間の計算 (パーセンタイル法)
            lower_ci = np.percentile(boot_indirect_effects, 2.5)
            upper_ci = np.percentile(boot_indirect_effects, 97.5)
            
            # === 結果の表示 ===
            print("\n" + "="*50)
            print(f"媒介分析結果 (Bootstrap samples={n_boot})")
            print(f"Model: {x_col} (X) -> {m_col} (M) -> {y_col} (Y)")
            if cov_vars: print(f"Controls: {cov_vars}")
            print("-" * 50)
            
            # パス係数の表示
            print(f"Path a  (X -> M)     : {a_obs:.4f}")
            print(f"Path b  (M -> Y)     : {b_obs:.4f}")
            print(f"Path c' (Direct X->Y): {c_prime_obs:.4f}")
            print("-" * 50)
            print(f"Total Effect (c)     : {tot_obs:.4f}")
            print(f"Indirect Effect (ab) : {ind_obs:.4f}")
            print(f"95% Bootstrap CI     : [{lower_ci:.4f}, {upper_ci:.4f}]")
            print("="*50)
            
            # 判定コメント
            if lower_ci > 0 or upper_ci < 0:
                print(">> 判定: 95%信頼区間がゼロを含まないため、間接効果は「有意」です。")
                print("   (媒介効果が存在することが示唆されます)")
            else:
                print(">> 判定: 信頼区間がゼロを含んでいるため、間接効果は「有意ではありません」。")

            # 結果保存
            save_path = os.path.join(OUTPUT_DIR, "mediation_result.txt")
            with open(save_path, "w") as f:
                f.write(f"Indirect Effect: {ind_obs}\nCI: [{lower_ci}, {upper_ci}]")
            
        except Exception as e:
            print(f"\n計算中にエラーが発生しました: {e}")

if __name__ == "__main__":
    analyzer = SocialAnalyzer(DATA_FILE, INDEX_FILE)
    
    while True:
        print("\n==============================")
        print("1. 基本統計量")
        print("2. グループ間比較 (t検定 / 分散分析 / ノンパラ)")
        print("3. 順序ロジスティック回帰分析") # <--- 追加
        print("4. 通常の回帰分析 (OLS)")
        print("6. 主成分分析 (PCA)")
        print("7. 信頼性分析 (α係数)")
        print("8. 媒介分析 (ブートストラップ法)") # <--- 追加
        print("0. 終了")
        print("==============================")
        
        c = input("選択: ")
        if c == '0': break
        elif c == '1': analyzer.run_basic_stats()
        elif c == '2': analyzer.run_group_comparison()
        elif c == '3': analyzer.run_ordinal_regression() # <--- 追加
        elif c == '4': analyzer.run_regression()
        elif c == '6': analyzer.run_pca()
        elif c == '7': analyzer.run_reliability()
        elif c == '8': analyzer.run_mediation_analysis()