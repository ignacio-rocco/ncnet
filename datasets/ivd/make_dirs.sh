while read -r path tail; do
  mkdir -p $path
done < dirs.txt
