# ailever modules
import options
from training import train
from evaluation import evaluation

def main(options):
    print(f'[AILEVER] The device "{options.device}" is selected!')
    train(options)
    evaluation(options)
    print(f'[AILEVER] Your experiment is successfully finished!')

if __name__ == "__main__":
    options = options.load()
    main(options)
