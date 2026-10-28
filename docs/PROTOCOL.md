# 固定した主実験条件

対象は著者公開experiment-springの調和振動子、seed0である。commit bcc362235dc623ffe48f22ccc22417e02e9803b4を読み取り監査し、データ・モデル・学習・評価を独立移植した。参照ファイルとSHA256はupstream_manifest.jsonに記録した。

H=q²+p²、速度(2p,−2q)。50軌道各30点、時間[0,3]。方向はsquare-uniform、半径一様[.1,1]、solve_ivp rtol1e−10、その他既定。clean教師速度から座標だけにstd.1雑音を加える。前25軌道train、後25軌道test。乱数順は方向2個、半径1個、q雑音30個、p雑音30個。

両モデル2→200→200→2、hidden tanh、最終biasなし、直交重み、hidden bias既定。HNNは第2出力をHとする。構築前seed resetで同初期重み、各41200パラメータ。float32、full batch、Adam lr.001/weight_decay.0001、betas(.9,.999)/eps1e−8、2001更新。最終モデルのみ、テストを見た調整なし。

著者notebookをソース順一回で実行した状態を主評価にする。可視化データによる乱数消費も維持。真軌道を先に積分し、t_spanとt_evalをin-place1.09倍してモデルを積分。比較セルは真time20/model21.8。続く15初期点集約では共有時間を引き継ぎ、model最後horizonは約79.406。初期値rand(2)*1.6−.8+.2*sign(x)、各100点。solve_ivp rtol1e−12、既定RK45/atol1e−6。

train/test誤差棒は全成分二乗誤差の不偏std/√750。energy誤差棒は15損失を連続3個ずつ5群に分けた群平均の母std/√5。真Hはq²+p²、energy MSEは数値真軌道とモデル軌道のH差の二乗平均。

isolated_calls_diagnosticは毎回新規のtime20/model21.8を使う別診断で、主結果を置換しない。保存notebookの飛び番号から手動再実行履歴は復元できない。本文のH/2・一様energy規約は公開実装と異なり、今回の実行へ混ぜない。

公開内容は公式主実行のコード・説明・生成図・集約だけに限定する。生配列・重み・個別軌道・本文条件補足・予備実験・後続研究は含めない。再実行時は合成データからローカル生成する。
