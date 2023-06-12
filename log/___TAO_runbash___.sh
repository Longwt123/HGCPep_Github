#### TAO  the bash for running programs ###
TODAY=$(date +"Today is %A, %d of %B, the local time is %r")
qianzhui="HGCPep_"
# 获取当前时间
logTime=$(date "+%m%d%H%M")  # 或 $(date "+%Y%m%d%H%M%S")

#    bash ___TAO_runbash___.sh

if read -t 120 -p "please enter the test content ->  $qianzhui" content
then
  echo "=============================try about $content ================================="
  echo "======== $TODAY ======="
  python /mnt/sdb/home/lwt/tao/HGCPep_new/src/main.py  > /mnt/sdb/home/lwt/tao/HGCPep_new/log/finetune_log/log_"$logTime"_"$qianzhui$content".log 2>&1 & echo $! > /mnt/sdb/home/lwt/tao/HGCPep_new/log/__PID__.txt
  echo ""
else
    echo "waiting time out ! ! !"
fi
