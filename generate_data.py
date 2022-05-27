import os, argparse
import numpy as np
import pandas as pd

def generate_graph_seq2seq_io_data(df, x_offsets, y_offsets, add_time_in_day=True, add_day_in_week=False, scaler=None):
    """
    Generating samples to
    # x: (epoch_size, input_length, num_nodes, input_dim)
    # y: (epoch_size, output_length, num_nodes, output_dim)
    """
    num_samples, num_nodes = df.shape
    data = np.expand_dims(df.values, axis=-1)
    feature_list = [data]
    if add_time_in_day:
        time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
        time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
        feature_list.append(time_in_day)
    if add_day_in_week:
        dow = df.index.dayofweek
        dow_tiled = np.tile(dow, [1, num_nodes, 1]).transpose((2, 1, 0))
        feature_list.append(dow_tiled)

    data = np.concatenate(feature_list, axis=-1)
    x, y = [], []
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets))) # Exclusive
    for t in range(min_t, max_t):
        x.append(data[t + x_offsets, ...])
        y.append(data[t + y_offsets, ...])
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)

    return x, y

def generate_train_val_test(args):
    seq_length_x, seq_length_y = args.seq_length_x, args.seq_length_y
    df = pd.read_hdf(args.input_data_dir)

    # 0 is the latest observed sample.
    x_offsets = np.sort(np.concatenate((np.arange(-(seq_length_x - 1), 1, 1), ))) # t(-11) -> t(0)
    # Predict the next one hour
    y_offsets = np.sort(np.arange(args.y_start, (seq_length_y + 1), 1)) # t(1) -> t(12)

    # x: (num_samples, input_length, num_nodes, input_dim)
    # y: (num_samples, output_length, num_nodes, output_dim)
    x, y = generate_graph_seq2seq_io_data(df, x_offsets=x_offsets, y_offsets=y_offsets, add_time_in_day=True, add_day_in_week=args.dow)

    # Write the data into npz file
    num_samples = x.shape[0]
    num_test = round(num_samples * 0.2)
    num_train = round(num_samples * 0.7)
    num_val = num_samples - num_train - num_test
    x_train, y_train = x[:num_train], y[:num_train]
    x_val, y_val     = (x[num_train: num_train+num_val], y[num_train: num_train+num_val])
    x_test, y_test   = x[-num_test:], y[-num_test:]
    for phase in ["train", "val", "test"]:
        _x, _y = locals()["x_" + phase], locals()["y_" + phase]
        print(phase, "x: ", _x.shape, "y: ", _y.shape)
        np.savez_compressed(os.path.join(args.output_data_dir, f"{phase}.npz"), \
                                        x=_x, 
                                        y=_y, 
                                        x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
                                        y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_data_dir", type=str, default="./data/METR-LA", help="Output directory")
    parser.add_argument("--input_data_dir", type=str, default="./data/metr-la.h5", help="Input directory")
    parser.add_argument("--seq_length_x", type=int, default=12, help="Sequence Length")
    parser.add_argument("--seq_length_y", type=int, default=12, help="Sequence Length")
    parser.add_argument("--y_start", type=int, default=1, help="Y pred start")
    parser.add_argument("--dow", action="store_true")

    args = parser.parse_args()
    if os.path.exists(args.output_data_dir):
        reply = str(input(f"{args.output_data_dir} is already existed. Overwrite it? (y/n)")).lower().strip()
        if reply[0] != "y": exit
    else:
        os.makedirs(args.output_data_dir)
    generate_train_val_test(args)