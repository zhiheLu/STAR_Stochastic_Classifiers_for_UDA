import svhn2mnist
import usps
import syn2gtrsb


def Generator(source, target):
    if source == 'usps' or target == 'usps':
        return usps.Feature()
    elif source == 'svhn':
        return svhn2mnist.Feature()
    elif source == 'synth':
        return syn2gtrsb.Feature()


def Classifier(
        source, 
        target, 
        num_classifiers_train=2, 
        num_classifiers_test=1, 
        constraint='softplus',
        init='kaiming_u', 
        use_init=False
    ):

    if source == 'usps' or target == 'usps':
        return usps.Predictor(
            num_classifiers_train, 
            num_classifiers_test, 
            constraint, 
            init, 
            use_init
        )

    if source == 'svhn':
        return svhn2mnist.Predictor(
            num_classifiers_train, 
            num_classifiers_test, 
            constraint, 
            init, 
            use_init
        )

    if source == 'synth':
        return syn2gtrsb.Predictor(
            num_classifiers_train, 
            num_classifiers_test, 
            constraint, 
            init, 
            use_init
        )

