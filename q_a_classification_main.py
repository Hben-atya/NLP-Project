import torch
from dataset import DatasetForVQA, QA_Type_Dataset, create_df, load_data
from config_parser import config_args
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoModelForSequenceClassification, \
    AutoTokenizer, TrainingArguments, Trainer, logging
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, top_k_accuracy_score
import sklearn.preprocessing as preprocessing
import pandas as pd


if __name__ == '__main__':

    def compute_metrics(pred):
        # Metric computation function. Creates precision, recall, f1, and accuracy metrics
        # for the Q and A classification. For use later in pytorch-lightning trainer.
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='micro')
        acc = accuracy_score(labels, preds)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }


    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Parse configuration args
    dataset_args = config_args['dataset']
    QA_model_args = config_args['QA_model']

    # Load data
    df = load_data(r"C:\Users\dekel\OneDrive\Masters\NLP\NLP_Course_Project\VQA_RAD_DATA\VQA_RAD Dataset Public.json")
    # Reorganize answers
    df['A'] = pd.Series(str(x).split('\t')[0].lower() for x in list(df['A']))
    # Make a label encoder to map answers to unique integers
    le = preprocessing.LabelEncoder()
    le.fit(list(df['A']))

    # Create the different dataframes
    train_df = create_df('VQA_RAD_DATA/train_vqa_rad.json', le)
    val_df = create_df('VQA_RAD_DATA/val_vqa_rad.json', le)
    test_df = create_df('VQA_RAD_DATA/test_vqa_rad.json', le)

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "mmoradi/Robust-Biomed-RoBERTa-TextClassification")

    # ========== Question ========== #
    # Create question encodings and labels for train-set
    Train_Q_Text = train_df['Q'].to_list()
    Train_Q_Text = [str(x) for x in Train_Q_Text]
    Train_Q_Encodings = tokenizer(Train_Q_Text, truncation=True, padding=True)
    Train_Q_Labels = train_df['Q_Type'].to_list()

    # Create question encodings and labels for validation-set
    Val_Q_Text = val_df['Q'].to_list()
    Val_Q_Text = [str(x) for x in Val_Q_Text]
    Val_Q_Encodings = tokenizer(Val_Q_Text, truncation=True, padding=True)
    Val_Q_Labels = val_df['Q_Type'].to_list()

    # Create question encodings and labels for test-set
    Test_Q_Text = test_df['Q'].to_list()
    Test_Q_Text = [str(x) for x in Test_Q_Text]
    Test_Q_Encodings = tokenizer(Test_Q_Text, truncation=True, padding=True)
    Test_Q_Labels = test_df['Q_Type'].to_list()

    # Number of unique question types
    num_Q_Type_labels = 11

    # ========== Answer ========== #
    # Create answer encodings and labels for train-set
    Train_A_Text = train_df['A'].to_list()
    Train_A_Text = [str(x) for x in Train_A_Text]
    Train_A_Encodings = tokenizer(Train_A_Text, truncation=True, padding=True)
    Train_A_Labels = train_df['A_Type'].to_list()

    # Create answer encodings and labels for validation-set
    Val_A_Text = val_df['A'].to_list()
    Val_A_Text = [str(x) for x in Val_A_Text]
    Val_A_Encodings = tokenizer(Val_A_Text, truncation=True, padding=True)
    Val_A_Labels = val_df['A_Type'].to_list()

    # Create answer encodings and labels for test-set
    Test_A_Text = test_df['A'].to_list()
    Test_A_Text = [str(x) for x in Test_A_Text]
    Test_A_Encodings = tokenizer(Test_A_Text, truncation=True, padding=True)
    Test_A_Labels = test_df['A_Type'].to_list()

    # Number of unique answer types
    num_A_Type_labels = 2

    # Currently file is set to train the answer-type classifier. To change, change the labels to X_Q_Labels and
    # the num_A_Type_labels to num_Q_type_labels

    # Create the different datasets
    trainset = QA_Type_Dataset(encodings=Train_Q_Encodings, labels=Train_A_Labels)
    valset = QA_Type_Dataset(encodings=Val_Q_Encodings, labels=Val_A_Labels)
    testset = QA_Type_Dataset(encodings=Test_Q_Encodings, labels=Test_A_Labels)

    # Initialize the model
    model = AutoModelForSequenceClassification.from_pretrained(
        "mmoradi/Robust-Biomed-RoBERTa-TextClassification",
        num_labels=num_A_Type_labels)

    # Initialize the training arguments
    training_args = TrainingArguments(
        output_dir="./QA_Logs/Q_Type_Logs",  # Logging dir
        learning_rate=2e-5,  # Optimizer lr
        per_device_train_batch_size=128,  # Training batch size
        per_device_eval_batch_size=128,  # Validation batch size
        num_train_epochs=25,  # Number of epochs
        weight_decay=0.01,  # Optimizer weight decay
        fp16=True,  # Mixed-precision training
        do_eval=True,  # Evaluate model
        eval_steps=10,  # Validate every X steps
        save_steps=10,  # Log every X steps
        load_best_model_at_end=True,  # Load best model at end of training for evaluation
        evaluation_strategy='steps',  # Validation strategy
        save_strategy='steps',  # Logging strategy
        metric_for_best_model='f1',  # Metric for best model
        log_level='info'  # Something like verbosity
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=trainset,
        eval_dataset=valset,
        compute_metrics=compute_metrics  # Metrics from the function above
    )

    # Train model
    trainer.train()

    # Evaluate model with test-set
    trainer.evaluate(eval_dataset=testset, metric_key_prefix='eval')
