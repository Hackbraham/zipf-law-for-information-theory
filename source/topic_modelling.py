import gensim
import pyLDAvis.gensim
from gensim.models import CoherenceModel
from gensim.corpora import Dictionary
import multiprocessing
import optuna
import matplotlib.pyplot as plt


def create_LDA_model(corpus, num_topics, id2word, passes):

    lda_model = gensim.models.LdaMulticore(corpus,
                                           num_topics=num_topics,
                                           id2word=id2word,
                                           passes=passes,
                                           workers=multiprocessing.cpu_count() - 1)
    return lda_model


def evaluate_coherence(model, texts: list[list[str]], dictionary: Dictionary, coherence: str) -> float:
    coherence_model_lda = CoherenceModel(model=model,
                                         texts=texts,
                                         dictionary=dictionary,
                                         coherence=coherence)
    return coherence_model_lda.get_coherence()


def find_best_LDA(corpus, id2word, passes, texts, coherence, n_trials = 3):

    def single_study(trial):
        num_topics = trial.suggest_int("num_topics", 2, 20)
        model = create_LDA_model(
            corpus=corpus,
            num_topics=num_topics,
            id2word=id2word,
            passes=passes
        )
        ev = evaluate_coherence(
            model=model,
            texts=texts,
            dictionary=id2word,
            coherence=coherence
        )
        return ev  # single float value


    study = optuna.create_study(
        direction="maximize",
        study_name="LDA-topic-modelling"
    )

    study.optimize(single_study, n_trials=n_trials)

    ev_metric = study.trials_dataframe()
    ev_metric.rename(columns={'value': 'coherence_score'}, inplace=True)

    print(f"Best trial: {study.best_params}")
    plot_LDA_search_results(ev_metric)

    return ev_metric


def plot_LDA_search_results(df):
    df = df.sort_values(by=['params_num_topics'], ascending=False)
    num_topics = df['params_num_topics']
    coherence = df['coherence_score']

    plt.figure(figsize=(8, 5))
    plt.plot(num_topics, coherence, marker='o', linestyle='-', color='royalblue')
    plt.title("LDA Hyperparameter Search: Coherence vs Num Topics")
    plt.xlabel("Number of Topics")
    plt.ylabel("Coherence Score")
    plt.grid(True, alpha=0.3)
    plt.legend(["Coherence Score"], loc='best')
    plt.show()


def plot_lda_vis(lda_model, corpus, dic):
    pyLDAvis.enable_notebook()
    vis = pyLDAvis.gensim.prepare(lda_model, corpus, dic)
    return vis