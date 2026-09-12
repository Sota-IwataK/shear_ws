# P1-03B-01B LocalHoldStateEstimator

判定: **P1-03B-01B Software PASS**。Hardware PASSではない。
最終結果: colcon build成功、colcon test-result **89 tests, 0 errors,
0 failures, 0 skipped**、既存863ファイルのSHA256一致。

## 確認したこと

実装前にshear_wsでpwd、git status --short --branch、git rev-parse HEAD、
find . -maxdepth 3 -type f | sort | head -300を実施。
HEAD=df29960a78ebdb8c20d61764f2183383771afcda、branch=main...origin/main。
開始時はdocs/とsrc/photon_bottle_identifier/のみ未追跡。tracked diffなし。
指定5語をrgで検索（build/install/log除外）し、holds_object、held_object_id、
selected_object_id、grasp_candidateの既存Core実装がないことを確認。
gripper_stateは既存制御/BT関連に存在するが本Coreと結合しない。

amir_irt_mroはament_python、Python、setuptools、pytest構成。
既存real_safety等と同様に同packageの独立moduleとして追加した。
新規package/framework/dependency、entry point、launchは追加していない。
repository内には別途のP1-03A/P1-03B-01A Freeze成果物が見つからなかったため、
ユーザー添付のP1-03B-01B文面に再掲されたFreeze仕様を実装基準とした。
runtime監査BLOCKEDは継続。この作業ではruntimeにアクセスしない。

## 実装したこと

`amir_irt_mro.local_hold_state_estimator`:

- immutable dataclassのHoldConfig、Signal、OptionalLoad、NormalizedHoldInputs、
  LocalHoldState、および状態を保持するLocalHoldStateEstimator。
- 入力timestamp_nsと各Signal.timestamp_nsは同じclock domainの非負整数ns。
  Coreは時刻を取得しない。float秒のConfigはDecimal経由で整数nsへ変換し、
  1 nsより細かいConfigを拒否。秒の浮動小数点減算やepsilon判定は使わない。
- 必須signalごとのmissing/invalid/stale/非有限値/型/未来・逆行stampを検査。
  負のdistanceもinvalid。selected IDは非負intまたはNone（選択なし）。
  adapterは外部の未選択sentinelをNoneへ正規化する。
- LoadDetectedはoptional。unavailable/invalid/staleはload evidenceのみを
  unavailableへ落とす。main conditionや全体validの必須条件にしない。
- 出力に主条件、各evidence、距離、actual position、入力age、stale flags、
  invalid理由、loadの状態、semantic transition時刻を保存する。
  dataclasses.asdictでstructured logへ変換可能。

`amir_irt_mro.local_hold_mock` はfreshなsynthetic入力の生成helper。
ROSをimportせずsleepせず、任意の整数nsを渡して試験できる。

## Freeze仕様との対応

| Freeze項目 | 実装 |
|---|---|
| 主条件 | close AND inclusive intermediate interval AND distance <= threshold |
| Hold成立 | 主条件が観測上連続hold_confirm_sec=0.20 |
| Close解除 | false連続close_release_sec=0.10 |
| Bottle離脱 | near=false連続bottle_release_sec=0.30 |
| ID commit | holds_object=true成立からheld_id_commit_sec=0.40 |
| Candidate latch | 観測済みfalse→true edgeでselected IDをlatch |
| 選択変更 | attempt中のcandidate/held IDに反映しない |
| Release | held IDとcandidateをclear |
| Invalid | valid=false、holds=false、両IDとdebounce timerを即時clear |
| Load | 補助のみ。Load-only/CLOSE-onlyでHoldを成立させない |
| ROS/TF境界 | Coreにtopic、frame、domain、RMW、IP、namespaceを持たせない |

### ID lifecycleと未規定部分の明示

1. 起動時とinvalid復旧直後は前回CloseIntentがunknown。
   最初からtrueなら観測した立上りではないのでcandidateを捏造しない。
   主条件からHoldは成立可能だがIDはNone。観測したfalse→trueで次回latchする。
2. Hold成立前にopenになればpending candidateを直ちにclear。
   Hold成立後の短いopen pulseは指定debounceを適用しcandidateを維持。
   その間の再closeで別selected IDに上書きしない。
3. Bottle離脱でreleaseした場合も両IDをclear。Closeがtrueのまま再度主条件が
   成立するとHoldは再確認可能だが、新しいedgeなしにIDは再latchしない。
4. commitは「holds_object成立状態が安定」の時間であり、release debounce内の
   一時的なevidence falseでholdsが変わらなければcommit timerも維持する。
5. FreezeでHold後の解除に指定されているのはClose解除とBottle離脱。
   有効なIntermediateStop=false単独では第三のreleaseルールを追加しない。
   その場合main_condition=falseをdebug出力する。gripperのinvalid/staleは
   別途即時fail-safe。上記挙動はテストで明示している。

### 時刻・鮮度契約

各必須age上限、optional load age上限、max_update_gap_secはConfig化、仮既定0.5秒。
age==上限は有効、1 ns超過でstale。同一stampは許容し時間を進めない。
未来/逆行stampはinvalid。update timestamp逆行時はinvalid結果を返して新epochへ
resetし、次の有効入力から再確認する。古いHold時間を復元しない。
update間隔がmax_update_gap_secを超えた場合も、観測していない期間を成立時間へ
算入せずinvalid/reset。Coreを呼ばない間に自律更新はできないため、adapterは
周期的に呼び、入力を受け取れない場合もmissing/staleとして渡す必要がある。

Adapterはactual feedbackの正本、単位、distanceの共通座標系、対象IDとの対応、
clock整合と取得鮮度を保証する責任を持つ。通信の受信時刻だけを取得鮮度と
同一視しない。Coreはこれらの物理的な正しさを検証できない。

## 使用例（Syntheticのみ）

```python
from amir_irt_mro.local_hold_mock import mock_inputs
from amir_irt_mro.local_hold_state_estimator import LocalHoldStateEstimator

estimator = LocalHoldStateEstimator()
estimator.update(mock_inputs(0, close=False, selected_id=3))
estimator.update(mock_inputs(10_000_000, close=True, selected_id=3))
state = estimator.update(mock_inputs(210_000_000, close=True, selected_id=4))
assert state.holds_object and state.grasp_candidate_object_id == 3
state = estimator.update(mock_inputs(610_000_000, close=True, selected_id=4))
assert state.held_object_id == 3
```

## T1〜T10結果

| Test | 内容 | 結果 |
|---|---|---|
| T1 | Openでvalid=true/holds=false | PASS |
| T2 | Empty close、intermediate/near不成立、load単独不成立 | PASS |
| T3 | 0.19秒false、0.20秒Hold | PASS |
| T4 | candidate=3のlatchとselected=4への変更拒否 | PASS |
| T5 | Hold後0.39秒未commit、0.40秒ID=3 | PASS |
| T6 | Openの0.10秒debounce、ID clear、次attemptの新ID | PASS |
| T7 | Bottle離脱0.30秒debounce、両ID clear | PASS |
| T8 | 0.20秒未満の反復pulseを拒否 | PASS |
| T9 | 必須各入力の欠落/invalid/stale/非有限値/未来/逆行 | PASS |
| T10 | Load unavailableでも正常Hold | PASS |

## 追加テスト結果

4つの時間条件それぞれの1 ns直前/一致/直後、閾値両端、短いClose/Bottle dropout、
optional load異常、update停止、再起動、invalid復旧、時計逆行後の再確認、
candidate中断、選択なし、不正ID、負距離、設定変更/不正Config/ゼロdelay、
入力age境界、同一時刻、transition timestampを検証。
pytestは全89ケースPASS（T1〜T10のparameterizedケースと追加試験を含む）。
これはソフトウェアのsynthetic試験結果であり実機検証ではない。

## Build結果

初回は隔離install先にamir_interfacesがなく失敗。原因はcolconのworkspace内
依存package探索。既存package/dependency構成を確認し、対象packageのみに
--base-pathsを限定したbuildが成功した。
underlayを読む一度の試行でlocal_setup.bashのパスを誤り、存在しないエラーが
出たため、実際のshare/amir_interfaces/local_setup.bashへ修正した。
production configや既存依存packageは変更していない。
追加ファイルflake8は指摘した改行を修正後exit 0。

最終実行コマンドとcolcon結果はevidence_01b/build.txt、colcon_test.txt、
colcon_test_result.txtに保存済み。対象Coreのtestを選択し、既存制御nodeや
repository全体のlint/testのPASSを主張しない。

```bash
source /opt/ros/humble/setup.bash
source install/amir_interfaces/share/amir_interfaces/local_setup.bash
colcon --log-base log/p1_03b_01b build \
  --base-paths src/amir_irt_mro --packages-select amir_irt_mro \
  --build-base build/p1_03b_01b --install-base install/p1_03b_01b
colcon --log-base log/p1_03b_01b test \
  --base-paths src/amir_irt_mro --packages-select amir_irt_mro \
  --build-base build/p1_03b_01b --install-base install/p1_03b_01b \
  --pytest-args test/test_local_hold_state_estimator.py
colcon test-result --test-result-base build/p1_03b_01b --verbose
PYTHONPATH=src/amir_irt_mro python3 -m pytest \
  src/amir_irt_mro/test/test_local_hold_state_estimator.py -q
```

試験実行時にHOLD_TRACE_PATHを設定し、各test名・人工timestamp付き入力・出力を
JSONLで記録。NaN/infの不正入力は文字列化してstrict JSONに保存。
JUnit XMLには試験実行時刻も含まれる。

install成果物についても確認。Python package内のlocal_setup.bashを試した際は
そのファイルがなくimportに失敗したため、生成ファイル構成を確認し、正しい
`source install/p1_03b_01b/local_setup.bash` に修正。
install配下からimportされたことをmodule.__file__で確認し、synthetic入力から
Hold成立・held ID=3のcommitまで成功（evidence_01b/installed_smoke.json）。

## 変更ファイル

- src/amir_irt_mro/amir_irt_mro/local_hold_state_estimator.py — Core/data/config。
- src/amir_irt_mro/amir_irt_mro/local_hold_mock.py — synthetic入力helper。
- src/amir_irt_mro/test/test_local_hold_state_estimator.py — T1〜T10と追加試験。
- docs/paper1/P1-03B-01/p1_03b_01b_implementation.md — 本記録。
- docs/paper1/P1-03B-01/evidence_01b/ — SHA256 baseline、試験trace、JUnit、build/test証拠。

既存trackedファイル全体と未追跡photon_bottle_identifierの実装前SHA256を保存。
終了時に比較し、863ファイルすべての無変更を検証済み
（evidence_01b/isolation_check.json）。既存control/servo/arm/base/
gripper/forward controllerのファイルは編集対象外。
git status / diff --check / diffを確認。新規未追跡ファイルは通常のgit diffに
出ないため、追加ファイルへのlint/testとSHA比較を併用する。

## 未確定事項

実機runtime正本・TF/Bottle frame・driver鮮度・校正値は未確定。
今回の対象はCoreのみでありHardware PASSではない。

## 実機復旧後に必要なCalibration

**requires hardware calibration**:
gripper_intermediate_min=0.2、max=0.8、bottle_near_threshold=0.1はmock専用仮値。
Coreはこれらを実機単位や正しい閾値と主張しない。
adapterの単位と共通frameの確認後、空close・物体把持・位置ノイズから閾値を校正。
各age上限とupdate gapも観測周期/取得時刻に合わせて検証する。
LoadDetected availabilityは別途確認し、なくても主判定を成立可能とする。

## 次工程

runtime正本監査を再開し、Topic binding、TF/frame・取得時刻・物体ID対応を確定。
actual feedbackとdistanceを正規化するAdapterを後日実装し、校正後に実機の
empty-close/real-grasp/release/dropout/fail-safeと再起動再現性を検証する。
実機検証完了まではHardware PASSにしない。
