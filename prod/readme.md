The script is designed to do the following:
1. Take in 1 single audio file of any length (e.g. 60secs)
2. Chunk it into 2 seconds (e.g. 60 secs will have 30 chunks)
3. Feed chunks sequentially into model to get an array of probabilities of deepfake (e.g. 60 secs will have an array of size 30)
4. Increase resolution back to seconds by repeating each value twice (60 secs will have an array of size 60)

Script Variables
file_path: path of where the audio file is located

Script Returns
It just prints out the probabilities in an numpy array. For you guys to decide how you want to integrate into the API or use it for the testbed.

To run script, type in:
python main.py {file_path}
e.g.
python main.py ./data/predict/DE_F_122523.flac
