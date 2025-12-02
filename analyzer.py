import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
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
        """グループ単位で変数を選択させるUI"""
        print(f"\n--- {message} ---")
        group_names = list(self.groups.keys())
        
        # グループ一覧を表示
        for i, g_name in enumerate(group_names):
            vars_in_group = self.groups[g_name]
            # データフレームに実際に存在する変数だけに絞る
            valid_vars = [v for v in vars_in_group if v in self.df.columns]
            print(f"[{i}] {g_name} (項目数: {len(valid_vars)})")
        
        # 個別変数の選択オプションも表示
        print(f"[{len(group_names)}] (個別の変数を手動で選ぶ)")

        while True:
            try:
                user_input = input("番号を選択してください: ")
                idx = int(user_input)
                
                # グループ選択
                if 0 <= idx < len(group_names):
                    selected_group = group_names[idx]
                    selected_vars = [v for v in self.groups[selected_group] if v in self.df.columns]
                    print(f"選択された変数群: {selected_vars}")
                    return selected_vars
                
                # 個別選択モード
                elif idx == len(group_names):
                    return self.get_column_selection_manual(multi)
                
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

# ==========================================
# メイン実行部
# ==========================================
if __name__ == "__main__":
    analyzer = SocialAnalyzer(DATA_FILE, INDEX_FILE)
    
    while True:
        print("\n==============================")
        print("1. 基本統計量")
        print("4. 回帰分析")
        print("6. 主成分分析 (PCA)")
        print("7. 信頼性分析 (α係数 + 平均値作成)")
        print("0. 終了")
        print("==============================")
        
        c = input("選択: ")
        if c == '0': break
        elif c == '1': analyzer.run_basic_stats()
        elif c == '4': analyzer.run_regression()
        elif c == '6': analyzer.run_pca()
        elif c == '7': analyzer.run_reliability()