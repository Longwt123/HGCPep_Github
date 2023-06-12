#### TAO  the bash for running programs ###
TODAY=$(date +"Today is %A, %d of %B, the local time is %r")
qianzhui="testTAO_"
name=${1}

#if read -t 120 -p "please enter the test program name (without '.py') ->  " name
#then
if read -t 120 -p "please enter the test content ->  $qianzhui" content
then
  #for content in "testTAO_splitData"
  #do
  echo "============================test $name.py about $content ============================="
  echo "======== $TODAY ======="
  #python hgnnp_on_RNA.py > __log_"$context"__.log 2>&1 & echo $! > __PID__.txt
  #python preprocess.py > __log_"$context"__.log 2>&1 & echo $! > __PID__.txt
  python "$name" > __log_"$qianzhui$content"__.log 2>&1 & echo $! > __PID__.txt
  echo ""
  #done
  else
    echo "waiting time out ! ! !"
fi
#else
#  echo "waiting time out ! ! !"
#fi
