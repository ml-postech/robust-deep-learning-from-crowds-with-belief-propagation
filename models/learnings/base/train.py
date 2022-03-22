def add_base_train_args(parser):
    parser.add_argument('--seed', type=int, default=7777)
    parser.add_argument('--n_epochs', type=int, default=200)
    parser.add_argument('--test_interval', type=int, default=200)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=1024)

