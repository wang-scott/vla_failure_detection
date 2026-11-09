#!/usr/bin/env bash
# act_mean 逐層掃描（seen = --split episode、unseen = --split task）+ act_prelogits，
# 再以各切分 α=0.1 早期偵測分數最高的層輸出逐 episode 分數與標註影片。
set -eo pipefail
cd "$(dirname "$0")"
export OUTPUTS_DIR=outputs_act
O=outputs_act/sweep_$(date +%m%d)
mkdir -p "$O"

declare -A SPLIT=([seen]=episode [unseen]=task)

# 1. 逐層掃描（兩個切分平行跑）
for s in seen unseen; do
  python online_failure_detector.py --feature act_mean --sweep --repeats 1 --split "${SPLIT[$s]}" \
    --xlsx "$O/sweep_act_mean_$s.xlsx" > "$O/sweep_act_mean_$s.log" 2>&1 &
done
wait

# 2. act_prelogits（只有一層）
for s in seen unseen; do
  python online_failure_detector.py --feature act_prelogits --layer 0 --split "${SPLIT[$s]}" \
    > "$O/act_prelogits_$s.log" 2>&1
done

# 3. 各切分最佳層 -> 逐 episode 分數 -> 標註影片（unseen 另畫一份與 seen 相同的 episode 以便比對）
for s in seen unseen; do
  L=$(grep -E "^act_mean layer +[0-9]+:" "$O/sweep_act_mean_$s.log" \
      | sed -E 's/^act_mean layer +([0-9]+):.*早期偵測分數=([0-9.]+).*/\2 \1/' | sort -nr | head -1 | cut -d' ' -f2)
  echo "$s 最佳層 = $L" | tee -a "$O/best_layers.txt"
  python export_episode_scores.py --out "$O/episode_scores_act_mean_${s}_L$L" --feature act_mean --layer "$L" \
    --split "${SPLIT[$s]}" > "$O/export_${s}_L$L.log" 2>&1
  extra=()
  [ "$s" = unseen ] && extra=(--same_as "$O/videos_seen/index.csv")
  python render_detection_videos.py --scores "$O/episode_scores_act_mean_${s}_L$L.csv" \
    --out_dir "$O/videos_$s" "${extra[@]}" > "$O/render_$s.log" 2>&1
done
echo SWEEP_DONE | tee -a "$O/best_layers.txt"
