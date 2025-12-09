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
                    for i in range(len(selected_vars)): # range() が必要です
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

    def _generate_filename(self, analysis_type, y_col, x_cols=None, m_cols=None):
        """
        ファイル名を自動生成するヘルパー
        形式: {分析種別}_Y_{従属変数}_X_{独立変数}_M_{媒介変数}.txt
        """
        # ファイル名に使えない文字を除去する関数
        def sanitize(name):
            return re.sub(r'[\\/*?:"<>|]', "", str(name))

        name = f"{analysis_type}_Y-{sanitize(y_col)}"
        
        if x_cols:
            if isinstance(x_cols, list):
                # リストの場合は連結するが、長すぎたら切り詰める
                x_str = "-".join([sanitize(x) for x in x_cols])
                if len(x_str) > 30: x_str = x_str[:30] + "..."
                name += f"_X-{x_str}"
            else:
                name += f"_X-{sanitize(x_cols)}"
        
        if m_cols:
            if isinstance(m_cols, list):
                m_str = "-".join([sanitize(m) for m in m_cols])
                if len(m_str) > 30: m_str = m_str[:30] + "..."
                name += f"_M-{m_str}"
            else:
                name += f"_M-{sanitize(m_cols)}"
                
        return name

    # --------------------------------------
    # 分析メソッド (グループ選択UIを使用するように変更)
    # --------------------------------------

    def run_subgroup_filtering(self):
        print("\n=== サブグループ抽出 (データフィルタリング) ===")
        print("特定の条件に合致するデータだけを残して、分析対象を絞り込みます。")
        print("例: 'intent_selfish_induction' が 1 の人だけで分析したい場合など")
        
        # フィルタに使う変数を選択
        print("\n>> フィルタリング条件に使う変数を選んでください")
        # 誘導変数(0/1)を選ぶことが多いので手動選択
        col = self.get_column_selection_manual(multi=False)
        if isinstance(col, list): col = col[0]
        
        # その変数のユニークな値を表示して、選択のヒントにする
        try:
            unique_vals = sorted(self.df[col].dropna().unique())
            print(f"\n変数 '{col}' に含まれる値: {unique_vals}")
        except:
            pass
        
        # 値を入力
        val_str = input(f"どの値を持つデータを残しますか？ (例: 1): ")
        
        try:
            # 入力値をデータ型に合わせて変換
            # データ内の型が数値なら数値に変換、そうでなければ文字列のまま
            if pd.api.types.is_numeric_dtype(self.df[col]):
                val = float(val_str)
                # 整数比較のための調整
                if val.is_integer():
                    target_val = int(val)
                else:
                    target_val = val
            else:
                target_val = val_str
            
            # フィルタリング実行 (条件に合うものだけを self.df に代入)
            initial_n = len(self.df)
            
            # 数値型でのフィルタリングは少し注意が必要（浮動小数の誤差など）
            # ここではシンプルに一致判定
            if pd.api.types.is_numeric_dtype(self.df[col]):
                self.df = self.df[self.df[col] == float(target_val)]
            else:
                self.df = self.df[self.df[col].astype(str) == str(target_val)]
            
            new_n = len(self.df)
            
            print(f"\n>> 抽出完了: N = {initial_n} -> {new_n}")
            print("これ以降のすべての分析は、このサブグループに対して行われます。")
            print("全データに戻す場合は、メニューから 'データの再読み込み(リセット)' を選んでください。")
            
        except Exception as e:
            print(f"エラー: フィルタリングに失敗しました。 ({e})")

    def reload_data(self):
        print("\n=== データの再読み込み (リセット) ===")
        print("フィルタリングを解除し、初期状態に戻します。")
        # __init__ を再度呼び出してリロードする
        # グローバル変数の DATA_FILE, INDEX_FILE を利用
        self.__init__(DATA_FILE, INDEX_FILE)
        print(">> データをリセットしました。")

    def run_standardization(self):
        print("\n=== 変数の標準化 (Z得点化) ===")
        print("変数を「平均 0、標準偏差 1」に変換し、新しい変数として保存します。")
        print("作成される変数名: z_元の変数名 (例: z_invest_A)")
        
        # 変数選択 (複数可)
        target_cols = self.get_group_selection("標準化したい変数を選んでください", multi=True)
        
        if not target_cols:
            return
        
        count = 0
        for col in target_cols:
            # 数値型かどうかチェック
            if pd.api.types.is_numeric_dtype(self.df[col]):
                # 計算 ( (x - mean) / std )
                mean_val = self.df[col].mean()
                std_val = self.df[col].std(ddof=1) # 不偏標準偏差
                
                if std_val == 0:
                    print(f"スキップ: {col} (標準偏差が0＝すべての値が同じ ため計算できません)")
                    continue
                
                new_col_name = f"z_{col}"
                self.df[new_col_name] = (self.df[col] - mean_val) / std_val
                
                print(f" -> 作成: {new_col_name} (Mean={mean_val:.2f}, Std={std_val:.2f})")
                count += 1
            else:
                print(f"スキップ: {col} (数値ではないため)")

        if count > 0:
            print(f"\n>> 計 {count} 個の変数を標準化しました。")
            # 念のため保存するか聞く
            if input("データをCSVに保存しますか？ (y/n): ").lower() == 'y':
                save_path = os.path.join(OUTPUT_DIR, "data_standardized.csv")
                self.df.to_csv(save_path, index=False, encoding='utf-8_sig')
                print(f"保存しました: {save_path}")
            
            print("これ以降の分析では、作成された 'z_変数名' を選択できます。")

    def run_basic_stats(self):
        print("\n=== 基本統計量 ===")
        numeric_df = self.df.select_dtypes(include=[np.number])
        stats = numeric_df.describe()
        print(stats)
        stats.to_csv(os.path.join(OUTPUT_DIR, "basic_stats.csv"))
        print("basic_stats.csv に保存しました。")

    def run_distribution_check(self):
        print("\n=== 変数の分布確認 (ヒストグラム & 歪度/尖度) ===")
        
        # 変数選択 (複数可)
        print(">> 分布を確認したい変数を選んでください。")
        target_cols = self.get_group_selection("変数を選択", multi=True)
        if not target_cols: return

        print(f"\n{len(target_cols)} 個の変数の分布を描画します...")
        
        for col in target_cols:
            data = self.df[col].dropna()
            
            # 統計量の計算
            skew = data.skew()
            kurt = data.kurt()
            
            # 描画
            plt.figure(figsize=(8, 6))
            sns.histplot(data, kde=True, bins=15, color='skyblue')
            plt.title(f"分布: {col}\nSkewness={skew:.2f}, Kurtosis={kurt:.2f}")
            plt.xlabel(col)
            plt.ylabel("Count")
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # 保存
            # ファイル名に使えない文字があれば置換するなどの配慮
            safe_col_name = col.replace("/", "_").replace(":", "")
            save_path = os.path.join(OUTPUT_DIR, f"dist_{safe_col_name}.png")
            plt.savefig(save_path)
            plt.close() # メモリ解放
            
            print(f" -> 保存: {save_path} (Skew: {skew:.2f}, Kurt: {kurt:.2f})")
        
        print("\n※ Skewness(歪度): 0なら左右対称。正なら左に偏り、負なら右に偏る。")
        print("※ Kurtosis(尖度): 0なら正規分布並み。正なら尖り、負なら平坦。")

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

        # 4. 結果の表示と保存 (修正箇所)
        if results:
            results_df = pd.DataFrame(results)
            print("\n【検定結果一覧】")
            cols = ["Variable", "Test_Type", "Statistic", "p_value", "Significance", "Effect_Size_Type", "Effect_Size"]
            print(results_df[cols])
            
            # --- ファイル名変更 ---
            base_name = "GroupComp_NonParam" if is_non_parametric else "GroupComp_Param"
            # グループ変数名(group_col)を含める
            filename = f"{base_name}_by_{group_col}.csv"
            
            save_path = os.path.join(OUTPUT_DIR, filename)
            results_df.to_csv(save_path, index=False, encoding='utf-8_sig')
            print(f"\n>> 結果を保存しました: {save_path}")
            print("   (Effect Size目安: d=0.2/0.5/0.8, r=0.1/0.3/0.5, eta2=0.01/0.06/0.14)")

    def run_distribution_check(self):
        print("\n=== 変数の分布確認 (ヒストグラム & 歪度/尖度) ===")
        
        # 変数選択 (複数可)
        print(">> 分布を確認したい変数を選んでください。")
        target_cols = self.get_group_selection("変数を選択", multi=True)
        if not target_cols: return

        print(f"\n{len(target_cols)} 個の変数の分布を描画します...")
        
        for col in target_cols:
            data = self.df[col].dropna()
            
            # 統計量の計算
            skew = data.skew()
            kurt = data.kurt()
            
            # 描画
            plt.figure(figsize=(8, 6))
            sns.histplot(data, kde=True, bins=15, color='skyblue')
            plt.title(f"分布: {col}\nSkewness={skew:.2f}, Kurtosis={kurt:.2f}")
            plt.xlabel(col)
            plt.ylabel("Count")
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # 保存
            # ファイル名に使えない文字があれば置換するなどの配慮
            safe_col_name = col.replace("/", "_").replace(":", "")
            save_path = os.path.join(OUTPUT_DIR, f"dist_{safe_col_name}.png")
            plt.savefig(save_path)
            plt.close() # メモリ解放
            
            print(f" -> 保存: {save_path} (Skew: {skew:.2f}, Kurt: {kurt:.2f})")
        
        print("\n※ Skewness(歪度): 0なら左右対称。正なら左に偏り、負なら右に偏る。")
        print("※ Kurtosis(尖度): 0なら正規分布並み。正なら尖り、負なら平坦。")

    def run_correlation_analysis(self):
        print("\n=== 相関分析 & ヒートマップ ===")
        
        # 変数選択
        print(">> 相関を見たい変数群を選んでください。")
        target_cols = []
        while True:
            selection = self.get_group_selection("追加する変数(群)を選んでください")
            if isinstance(selection, list):
                target_cols.extend(selection)
            else:
                target_cols.append(selection)
            
            if len(target_cols) >= 2:
                cont = input(f"現在 {len(target_cols)} 個選択中。さらに追加しますか？ (y/n): ")
                if cont.lower() != 'y':
                    break
            else:
                print("相関を計算するには少なくとも2つの変数が必要です。")
        
        # 重複除去
        target_cols = list(set(target_cols))
        
        # 相関係数の種類選択
        print("\n>> 計算手法を選んでください")
        print("[1] Pearson (積率相関) : 連続変数、正規分布向け")
        print("[2] Spearman (順位相関): 順序変数、非正規分布向け ★順序尺度ならこちら")
        method = 'pearson' if input("選択 (1/2): ") != '2' else 'spearman'
        
        # 計算
        corr_matrix = self.df[target_cols].corr(method=method)
        
        # 1. CSV保存
        csv_path = os.path.join(OUTPUT_DIR, f"correlation_matrix_{method}.csv")
        corr_matrix.to_csv(csv_path, encoding='utf-8_sig')
        print(f"\n>> 相関行列を保存しました: {csv_path}")
        
        # 2. ヒートマップ描画と保存
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', 
                    vmin=-1, vmax=1, center=0, square=True)
        plt.title(f"Correlation Heatmap ({method.capitalize()})")
        
        img_path = os.path.join(OUTPUT_DIR, f"correlation_heatmap_{method}.png")
        plt.savefig(img_path, bbox_inches='tight')
        plt.close()
        print(f">> ヒートマップ画像を保存しました: {img_path}")

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

    def _ask_add_constant(self):
        """定数項(切片)を追加するかユーザーに尋ねる"""
        print("\n>> 定数項 (Intercept/Constant) をモデルに追加しますか？")
        print("   y: 追加する (推奨 / Default) -> y = ax + b")
        print("   n: 追加しない (原点を通る)   -> y = ax")
        choice = input("選択 (y/n): ").lower()
        return choice != 'n'

    def run_regression(self):
        print("\n=== 回帰分析 (OLS) ===")
        # 従属変数は単一選択
        print(">> 従属変数(Y)を選びます。")
        y = self.get_column_selection_manual(multi=False)
        
        print(">> 独立変数(X)を選びます。グループ選択も可能です。")
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
        
        x_vars = list(set(x_vars))
        print(f"独立変数リスト: {x_vars}")

        # 定数項の確認
        add_const = self._ask_add_constant()

        data_reg = self.df[[y] + x_vars].dropna()
        Y_val = data_reg[y]
        X_val = data_reg[x_vars]
        
        if add_const:
            X_val = sm.add_constant(X_val)
        
        model = sm.OLS(Y_val, X_val).fit()
        print(model.summary())
        
        # ファイル名変更
        fname = self._generate_filename("OLS", y, x_cols=x_vars) + ".txt"
        save_path = os.path.join(OUTPUT_DIR, fname)
        
        with open(save_path, "w", encoding='utf-8') as f:
            f.write(model.summary().as_text())
        print(f">> 結果を保存しました: {save_path}")

    def run_ordinal_regression(self):
        print("\n=== 順序ロジスティック回帰分析 ===")
        print("注: リッカート尺度などの「順序変数」を従属変数(Y)とする場合に使用します。")
        
        # 1. 従属変数(Y)の選択
        print(">> 従属変数(Y)を選んでください（順序尺度である必要があります）。")
        y_col = self.get_column_selection_manual(multi=False)
        if isinstance(y_col, list): y_col = y_col[0]
        
        # 2. 独立変数(X)の選択
        print("\n>> 独立変数(X)を選んでください。")
        x_vars = []
        while True:
            selection = self.get_group_selection("追加する変数(群)を選んでください")
            x_vars.extend(selection)
            cont = input("さらに変数を追加しますか？ (y/n): ")
            if cont.lower() != 'y': break
        
        x_vars = list(set(x_vars))
        if y_col in x_vars: x_vars.remove(y_col)
            
        print(f"\nモデル構築中: {y_col} ~ {x_vars}")
        
        # 定数項の確認
        add_const = self._ask_add_constant()
        if add_const:
            print("※注: 順序ロジスティック回帰では、通常は定数項(切片)を含めません（閾値に含まれるため）。")
            print("      意図的に含める場合のみ、そのまま続行してください。")

        # データ準備
        data_reg = self.df[[y_col] + x_vars].dropna()
        if len(data_reg) < 5:
            print("エラー: 有効なデータ件数が少なすぎます。")
            return

        try:
            exog = data_reg[x_vars]
            if add_const:
                exog = sm.add_constant(exog)

            # モデル定義
            model = OrderedModel(data_reg[y_col], exog, distr='logit')
            res = model.fit(method='bfgs', disp=False)
            
            print("\n" + "="*40)
            print(res.summary())
            print("="*40)
            
            # 結果保存
            fname = self._generate_filename("OrdReg", y_col, x_cols=x_vars) + ".txt"
            save_file = os.path.join(OUTPUT_DIR, fname)
            with open(save_file, "w", encoding='utf-8') as f:
                f.write(res.summary().as_text())
            print(f">> 結果を保存しました: {save_file}")
            
            # オッズ比
            print("\n【参考: オッズ比 (Odds Ratios)】")
            params = res.params
            conf = res.conf_int()
            conf['Odds Ratio'] = params
            conf.columns = ['2.5%', '97.5%', 'Odds Ratio']
            out = np.exp(conf)
            print(out[['Odds Ratio', '2.5%', '97.5%']])

        except Exception as e:
            print(f"\n計算エラーが発生しました: {e}")

    def run_mediation_analysis(self):
        print("\n=== ハイブリッド媒介分析 (Y:順序尺度 / M:連続尺度) ===")
        print("------------------------------------------")
        print("[1] 単純媒介分析 (Model 4): X -> M -> Y")
        print("[2] 系列媒介分析 (Model 6): X -> M1 -> M2 -> Y")
        print("------------------------------------------")
        
        mode = input("分析モデルを選択してください (1 or 2): ")
        
        # ここで定数項の設定を一括して聞く
        add_const = self._ask_add_constant()
        
        if mode == '1':
            self._run_simple_mediation_hybrid(add_const)
        elif mode == '2':
            self._run_serial_mediation_hybrid(add_const)
        else:
            print("無効な選択です。メニューに戻ります。")

    def _run_simple_mediation_hybrid(self, add_const):
        """単純媒介分析 (Model 4) の実行部"""
        print("\n--- Model 4: 単純媒介分析 (X -> M -> Y) ---")
        
        # 変数選択 (省略せず記述)
        print("\n>> 1. 従属変数 (Y: 順序尺度) を選んでください")
        y_col = self.get_column_selection_manual(multi=False)
        if isinstance(y_col, list): y_col = y_col[0]

        print("\n>> 2. 独立変数 (X) を選んでください")
        x_col = self.get_column_selection_manual(multi=False)
        if isinstance(x_col, list): x_col = x_col[0]
        
        print("\n>> 3. 媒介変数 (M: 連続尺度と仮定) を選んでください")
        m_col = self.get_column_selection_manual(multi=False)
        if isinstance(m_col, list): m_col = m_col[0]

        cov_vars = self._select_covariates([y_col, x_col, m_col])
        
        target_cols = [y_col, x_col, m_col] + cov_vars
        df_model = self.df[target_cols].apply(pd.to_numeric, errors='coerce').dropna()
        
        if len(df_model) < 20:
            print("エラー: サンプルサイズが不足しています。")
            return

        print(f"\n分析対象データ数: N = {len(df_model)}")
        print("ブートストラップ検定を実行中... (Model 4)")

        # 内部関数: 推定ロジック (add_constを使用)
        def fit_model4(data):
            # Path a (X -> M): OLS
            X_a = data[[x_col] + cov_vars]
            if add_const: X_a = sm.add_constant(X_a)
            
            model_a = sm.OLS(data[m_col], X_a).fit()
            a = model_a.params[x_col]
            
            # Path b, c' (X, M -> Y): Ordered Logit
            exog_b = data[[x_col, m_col] + cov_vars]
            if add_const: exog_b = sm.add_constant(exog_b)
            
            model_b = OrderedModel(data[y_col], exog_b, distr='logit')
            res_b = model_b.fit(method='bfgs', disp=False, maxiter=50)
            
            b = res_b.params[m_col]
            c_prime = res_b.params[x_col]
            
            return {'a': a, 'b': b, 'c_prime': c_prime, 'ind': a * b}

        # 実行と結果表示
        self._execute_bootstrap_and_report(
            df_model, fit_model4, "Model4", ['ind'], 
            y_col, x_col, [m_col]
        )

    def _run_serial_mediation_hybrid(self, add_const):
        """系列媒介分析 (Model 6) の実行部"""
        print("\n--- Model 6: 系列媒介分析 (X -> M1 -> M2 -> Y) ---")
        
        # 変数選択
        print("\n>> 1. 従属変数 (Y: 順序尺度) を選んでください")
        y_col = self.get_column_selection_manual(multi=False)
        if isinstance(y_col, list): y_col = y_col[0]

        print("\n>> 2. 独立変数 (X) を選んでください")
        x_col = self.get_column_selection_manual(multi=False)
        if isinstance(x_col, list): x_col = x_col[0]
        
        print("\n>> 3. 第1媒介変数 (M1: 連続と仮定) を選んでください")
        m1_col = self.get_column_selection_manual(multi=False)
        if isinstance(m1_col, list): m1_col = m1_col[0]
        
        print("\n>> 4. 第2媒介変数 (M2: 連続と仮定) を選んでください")
        m2_col = self.get_column_selection_manual(multi=False)
        if isinstance(m2_col, list): m2_col = m2_col[0]

        cov_vars = self._select_covariates([y_col, x_col, m1_col, m2_col])
        
        target_cols = [y_col, x_col, m1_col, m2_col] + cov_vars
        df_model = self.df[target_cols].apply(pd.to_numeric, errors='coerce').dropna()
        
        if len(df_model) < 20:
            print("エラー: サンプルサイズが不足しています。")
            return

        print(f"\n分析対象データ数: N = {len(df_model)}")
        print("ブートストラップ検定を実行中... (Model 6)")

        # 内部関数: 推定ロジック (add_constを使用)
        def fit_model6(data):
            # Step 1: M1 ~ X (OLS)
            X_1 = data[[x_col] + cov_vars]
            if add_const: X_1 = sm.add_constant(X_1)
            
            m1_model = sm.OLS(data[m1_col], X_1).fit()
            a1 = m1_model.params[x_col]
            
            # Step 2: M2 ~ X + M1 (OLS)
            X_2 = data[[x_col, m1_col] + cov_vars]
            if add_const: X_2 = sm.add_constant(X_2)
            
            m2_model = sm.OLS(data[m2_col], X_2).fit()
            a2 = m2_model.params[x_col]
            d21 = m2_model.params[m1_col]
            
            # Step 3: Y ~ X + M1 + M2 (Ordered Logit)
            exog_y = data[[x_col, m1_col, m2_col] + cov_vars]
            if add_const: exog_y = sm.add_constant(exog_y)
            
            y_model = OrderedModel(data[y_col], exog_y, distr='logit')
            y_res = y_model.fit(method='bfgs', disp=False, maxiter=50)
            
            c_prime = y_res.params[x_col]
            b1 = y_res.params[m1_col]
            b2 = y_res.params[m2_col]
            
            ind1 = a1 * b1
            ind2 = a2 * b2
            ind3 = a1 * d21 * b2
            total_ind = ind1 + ind2 + ind3
            
            return {
                'ind1': ind1, 'ind2': ind2, 'ind3': ind3, 'total_ind': total_ind,
                'a1': a1, 'a2': a2, 'd21': d21, 'b1': b1, 'b2': b2, 'c_prime': c_prime
            }

        self._execute_bootstrap_and_report(
            df_model, fit_model6, "Model6", 
            ['ind1', 'ind2', 'ind3', 'total_ind'],
            y_col, x_col, [m1_col, m2_col]
        )

    def _select_covariates(self, exclude_cols):
        """統制変数の選択ヘルパー"""
        print("\n>> 統制変数 (Covariates) を追加しますか？")
        cov_vars = []
        if input("追加する場合は 'y' を入力: ").lower() == 'y':
            cov_vars = self.get_group_selection("統制変数を選んでください")
        # 重複除外
        used_cols = set(exclude_cols)
        return [c for c in cov_vars if c not in used_cols]

    def _execute_bootstrap_and_report(self, df, fit_func, title, effect_keys, y_name, x_name, m_names):
        """ブートストラップ実行と結果レポートの共通処理"""
        n_boot = 2000
        results_boot = {k: [] for k in effect_keys}
        success_count = 0
        
        # 1. 観測データでの推定
        try:
            obs_res = fit_func(df)
        except Exception as e:
            print(f"初期分析エラー: {e}")
            return

        # 2. ブートストラップ
        for i in range(n_boot):
            sample = resample(df, replace=True, n_samples=len(df))
            try:
                res = fit_func(sample)
                for k in effect_keys:
                    results_boot[k].append(res[k])
                success_count += 1
            except:
                pass # 収束しない場合はスキップ
            
            if (i+1) % 500 == 0:
                print(f"... {i+1} iterations")

        if success_count < 100:
            print("エラー: ブートストラップの成功回数が少なすぎます。")
            return

        # 3. 結果表示
        print(f"\n{'='*60}")
        print(f"【{title} 結果】 (N={len(df)}, Boot={success_count})")
        print(f"{'-'*60}")
        
        # 観測変数のパス係数などを表示（簡易版）
        print(">> Point Estimates (Path Coefficients):")
        for k, v in obs_res.items():
            if k not in effect_keys: # パス係数のみ表示
                print(f"   {k}: {v:.4f}")
        
        print(f"{'-'*60}")
        print(">> Indirect Effects (Bootstrap 95% CI):")
        
        save_lines = [f"{title} Results\n"]
        
        for k in effect_keys:
            vals = results_boot[k]
            lower = np.percentile(vals, 2.5)
            upper = np.percentile(vals, 97.5)
            pt_est = obs_res[k]
            
            sig = "*" if (lower > 0 or upper < 0) else "n.s."
            
            # 表示名の整形
            if k == 'ind': label = "Indirect (X->M->Y)"
            elif k == 'ind1': label = "Ind1 (X->M1->Y)"
            elif k == 'ind2': label = "Ind2 (X->M2->Y)"
            elif k == 'ind3': label = "Ind3 (X->M1->M2->Y)"
            elif k == 'total_ind': label = "Total Indirect"
            else: label = k
            
            print(f" {label:<20} : {pt_est:.4f}  [{lower:.4f}, {upper:.4f}] {sig}")
            save_lines.append(f"{label}: {pt_est:.4f} CI[{lower:.4f}, {upper:.4f}]\n")

        print(f"{'='*60}")
        
        # 保存
        fname = self._generate_filename(f"Mediation_{title}", y_name, x_name, m_names) + ".txt"
        save_path = os.path.join(OUTPUT_DIR, fname)
        
        with open(save_path, "w", encoding='utf-8') as f:
            f.writelines(save_lines)
        print(f"結果を保存しました: {save_path}")

if __name__ == "__main__":
    analyzer = SocialAnalyzer(DATA_FILE, INDEX_FILE)
    
    while True:
        print("\n==============================")
        print("0. 終了")
        print("1. サブグループ抽出 (フィルタリング)")
        print("2. データの再読み込み (リセット)")
        print("3. 変数の標準化 (Z得点化)")
        print("4. 基本統計量")
        print("5. 変数の分布確認 (ヒストグラム)")
        print("6. 信頼性分析 (α係数)")
        print("7. 相関分析 & ヒートマップ")
        print("8. グループ間比較 (t検定 / 分散分析 / ノンパラ)")
        print("9. 主成分分析 (PCA)")
        print("10. 通常の回帰分析 (OLS)")
        print("11. 順序ロジスティック回帰分析")
        print("12. 媒介分析 (Model 4 / Model 6)")
        print("==============================")
        
        c = input("選択: ")
        if c == '0': break
        elif c == '1': analyzer.run_subgroup_filtering()
        elif c == '2': analyzer.reload_data()
        elif c == '3': analyzer.run_standardization()
        elif c == '4': analyzer.run_basic_stats()
        elif c == '5': analyzer.run_distribution_check()
        elif c == '6': analyzer.run_reliability()
        elif c == '7': analyzer.run_correlation_analysis()
        elif c == '8': analyzer.run_group_comparison()
        elif c == '9': analyzer.run_pca()
        elif c == '10': analyzer.run_regression()
        elif c == '11': analyzer.run_ordinal_regression()
        elif c == '12': analyzer.run_mediation_analysis()