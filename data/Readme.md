## Data

This part of code will collect training data for RL algorithm.

## Real Data

Run `sudo python3 real_data.py` and the real throughput will be written in `net_data.txt`. To get more accurate data, you can play some videos or watch live shows with your client, then run this script.

## Synthetic Data

Run `python3 synthetic.py` to get some synthetic data according to normal distribution, both stable and unstable throughput are included. You can change the parameters of normal distribution to get data in different range.

## Get Dataset

If you need a large dataset to train your model, maybe you can try the following commands.

1) FCC broadband dataset
```
wget http://data.fcc.gov/download/measuring-broadband-america/2016/data-raw-2016-jun.tar.gz
tar -xvzf data-raw-2016-jun.tar.gz -C fcc
```

2) Norway HSDPA bandwidth logs
`wget -r --no-parent --reject "index.html*" http://home.ifi.uio.no/paalh/dataset/hsdpa-tcp-logs/`

3) Belgium 4G/LTE bandwidth logs (bonus)
```
wget http://users.ugent.be/~jvdrhoof/dataset-4g/logs/logs_all.zip
unzip logs_all.zip -d belgium
```
