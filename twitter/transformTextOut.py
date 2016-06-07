import io

out = """
FEATURES: uni-bi-extra-mpqa-senti
CLASSIFIERS:
Polar: multiNB/60pct
Sentiment: multiNB/60pct
Test accuracy: 0.633
Precision for neg: 0.739
Recall for neg: 0.559
F measure for neg: 0.637
Precision for neu: 0.706
Recall for neu: 0.518
F measure for neu: 0.598
Precision for pos: 0.550
Recall for pos: 0.791
F measure for pos: 0.649



CLASSIFIERS:
Polar: multiNB/60pct
Sentiment: multiNB/70pct
Test accuracy: 0.635
Precision for neg: 0.720
Recall for neg: 0.582
F measure for neg: 0.644
Precision for neu: 0.706
Recall for neu: 0.518
F measure for neu: 0.598
Precision for pos: 0.557
Recall for pos: 0.775
F measure for pos: 0.648



CLASSIFIERS:
Polar: multiNB/60pct
Sentiment: multiNB/80pct
Test accuracy: 0.647
Precision for neg: 0.721
Recall for neg: 0.627
F measure for neg: 0.671
Precision for neu: 0.706
Recall for neu: 0.518
F measure for neu: 0.598
Precision for pos: 0.574
Recall for pos: 0.764
F measure for pos: 0.656



CLASSIFIERS:
Polar: multiNB/60pct
Sentiment: logreg/60pct
Test accuracy: 0.657
Precision for neg: 0.656
Recall for neg: 0.712
F measure for neg: 0.683
Precision for neu: 0.706
Recall for neu: 0.518
F measure for neu: 0.598
Precision for pos: 0.632
Recall for pos: 0.709
F measure for pos: 0.668



CLASSIFIERS:
Polar: multiNB/60pct
Sentiment: logreg/70pct
Test accuracy: 0.653
Precision for neg: 0.643
Recall for neg: 0.723
F measure for neg: 0.681
Precision for neu: 0.706
Recall for neu: 0.518
F measure for neu: 0.598
Precision for pos: 0.635
Recall for pos: 0.687
F measure for pos: 0.660



CLASSIFIERS:
Polar: multiNB/60pct
Sentiment: logreg/80pct
Test accuracy: 0.655
Precision for neg: 0.637
Recall for neg: 0.734
F measure for neg: 0.682
Precision for neu: 0.706
Recall for neu: 0.518
F measure for neu: 0.598
Precision for pos: 0.646
Recall for pos: 0.681
F measure for pos: 0.663



CLASSIFIERS:
Polar: multiNB/70pct
Sentiment: multiNB/60pct
Test accuracy: 0.633
Precision for neg: 0.739
Recall for neg: 0.559
F measure for neg: 0.637
Precision for neu: 0.688
Recall for neu: 0.540
F measure for neu: 0.605
Precision for pos: 0.553
Recall for pos: 0.775
F measure for pos: 0.645



CLASSIFIERS:
Polar: multiNB/70pct
Sentiment: multiNB/70pct
Test accuracy: 0.635
Precision for neg: 0.720
Recall for neg: 0.582
F measure for neg: 0.644
Precision for neu: 0.688
Recall for neu: 0.540
F measure for neu: 0.605
Precision for pos: 0.561
Recall for pos: 0.758
F measure for pos: 0.645



CLASSIFIERS:
Polar: multiNB/70pct
Sentiment: multiNB/80pct
Test accuracy: 0.645
Precision for neg: 0.719
Recall for neg: 0.621
F measure for neg: 0.667
Precision for neu: 0.688
Recall for neu: 0.540
F measure for neu: 0.605
Precision for pos: 0.576
Recall for pos: 0.747
F measure for pos: 0.651



CLASSIFIERS:
Polar: multiNB/70pct
Sentiment: logreg/60pct
Test accuracy: 0.657
Precision for neg: 0.663
Recall for neg: 0.712
F measure for neg: 0.687
Precision for neu: 0.688
Recall for neu: 0.540
F measure for neu: 0.605
Precision for pos: 0.633
Recall for pos: 0.692
F measure for pos: 0.661



CLASSIFIERS:
Polar: multiNB/70pct
Sentiment: logreg/70pct
Test accuracy: 0.651
Precision for neg: 0.648
Recall for neg: 0.718
F measure for neg: 0.681
Precision for neu: 0.688
Recall for neu: 0.540
F measure for neu: 0.605
Precision for pos: 0.632
Recall for pos: 0.670
F measure for pos: 0.651



CLASSIFIERS:
Polar: multiNB/70pct
Sentiment: logreg/80pct
Test accuracy: 0.653
Precision for neg: 0.642
Recall for neg: 0.729
F measure for neg: 0.683
Precision for neu: 0.688
Recall for neu: 0.540
F measure for neu: 0.605
Precision for pos: 0.644
Recall for pos: 0.665
F measure for pos: 0.654



CLASSIFIERS:
Polar: multiNB/80pct
Sentiment: multiNB/60pct
Test accuracy: 0.629
Precision for neg: 0.739
Recall for neg: 0.559
F measure for neg: 0.637
Precision for neu: 0.670
Recall for neu: 0.540
F measure for neu: 0.598
Precision for pos: 0.552
Recall for pos: 0.764
F measure for pos: 0.641



CLASSIFIERS:
Polar: multiNB/80pct
Sentiment: multiNB/70pct
Test accuracy: 0.631
Precision for neg: 0.720
Recall for neg: 0.582
F measure for neg: 0.644
Precision for neu: 0.670
Recall for neu: 0.540
F measure for neu: 0.598
Precision for pos: 0.560
Recall for pos: 0.747
F measure for pos: 0.640



CLASSIFIERS:
Polar: multiNB/80pct
Sentiment: multiNB/80pct
Test accuracy: 0.641
Precision for neg: 0.719
Recall for neg: 0.621
F measure for neg: 0.667
Precision for neu: 0.670
Recall for neu: 0.540
F measure for neu: 0.598
Precision for pos: 0.575
Recall for pos: 0.736
F measure for pos: 0.646



CLASSIFIERS:
Polar: multiNB/80pct
Sentiment: logreg/60pct
Test accuracy: 0.651
Precision for neg: 0.661
Recall for neg: 0.706
F measure for neg: 0.683
Precision for neu: 0.670
Recall for neu: 0.540
F measure for neu: 0.598
Precision for pos: 0.629
Recall for pos: 0.681
F measure for pos: 0.654



CLASSIFIERS:
Polar: multiNB/80pct
Sentiment: logreg/70pct
Test accuracy: 0.645
Precision for neg: 0.646
Recall for neg: 0.712
F measure for neg: 0.677
Precision for neu: 0.670
Recall for neu: 0.540
F measure for neu: 0.598
Precision for pos: 0.628
Recall for pos: 0.659
F measure for pos: 0.643



CLASSIFIERS:
Polar: multiNB/80pct
Sentiment: logreg/80pct
Test accuracy: 0.647
Precision for neg: 0.640
Recall for neg: 0.723
F measure for neg: 0.679
Precision for neu: 0.670
Recall for neu: 0.540
F measure for neu: 0.598
Precision for pos: 0.640
Recall for pos: 0.654
F measure for pos: 0.647



CLASSIFIERS:
Polar: logreg/60pct
Sentiment: multiNB/60pct
Test accuracy: 0.618
Precision for neg: 0.762
Recall for neg: 0.542
F measure for neg: 0.634
Precision for neu: 0.653
Recall for neu: 0.475
F measure for neu: 0.550
Precision for pos: 0.539
Recall for pos: 0.802
F measure for pos: 0.645



CLASSIFIERS:
Polar: logreg/60pct
Sentiment: multiNB/70pct
Test accuracy: 0.622
Precision for neg: 0.743
Recall for neg: 0.571
F measure for neg: 0.645
Precision for neu: 0.653
Recall for neu: 0.475
F measure for neu: 0.550
Precision for pos: 0.548
Recall for pos: 0.786
F measure for pos: 0.646



CLASSIFIERS:
Polar: logreg/60pct
Sentiment: multiNB/80pct
Test accuracy: 0.633
Precision for neg: 0.740
Recall for neg: 0.610
F measure for neg: 0.669
Precision for neu: 0.653
Recall for neu: 0.475
F measure for neu: 0.550
Precision for pos: 0.562
Recall for pos: 0.775
F measure for pos: 0.651



CLASSIFIERS:
Polar: logreg/60pct
Sentiment: logreg/60pct
Test accuracy: 0.643
Precision for neg: 0.661
Recall for neg: 0.706
F measure for neg: 0.683
Precision for neu: 0.653
Recall for neu: 0.475
F measure for neu: 0.550
Precision for pos: 0.620
Recall for pos: 0.709
F measure for pos: 0.662



CLASSIFIERS:
Polar: logreg/60pct
Sentiment: logreg/70pct
Test accuracy: 0.637
Precision for neg: 0.649
Recall for neg: 0.712
F measure for neg: 0.679
Precision for neu: 0.653
Recall for neu: 0.475
F measure for neu: 0.550
Precision for pos: 0.616
Recall for pos: 0.687
F measure for pos: 0.649



CLASSIFIERS:
Polar: logreg/60pct
Sentiment: logreg/80pct
Test accuracy: 0.639
Precision for neg: 0.637
Recall for neg: 0.723
F measure for neg: 0.677
Precision for neu: 0.653
Recall for neu: 0.475
F measure for neu: 0.550
Precision for pos: 0.633
Recall for pos: 0.681
F measure for pos: 0.656



CLASSIFIERS:
Polar: logreg/70pct
Sentiment: multiNB/60pct
Test accuracy: 0.624
Precision for neg: 0.760
Recall for neg: 0.537
F measure for neg: 0.629
Precision for neu: 0.654
Recall for neu: 0.504
F measure for neu: 0.569
Precision for pos: 0.549
Recall for pos: 0.802
F measure for pos: 0.652



CLASSIFIERS:
Polar: logreg/70pct
Sentiment: multiNB/70pct
Test accuracy: 0.629
Precision for neg: 0.746
Recall for neg: 0.565
F measure for neg: 0.643
Precision for neu: 0.654
Recall for neu: 0.504
F measure for neu: 0.569
Precision for pos: 0.556
Recall for pos: 0.786
F measure for pos: 0.651



CLASSIFIERS:
Polar: logreg/70pct
Sentiment: multiNB/80pct
Test accuracy: 0.639
Precision for neg: 0.743
Recall for neg: 0.605
F measure for neg: 0.667
Precision for neu: 0.654
Recall for neu: 0.504
F measure for neu: 0.569
Precision for pos: 0.571
Recall for pos: 0.775
F measure for pos: 0.657



CLASSIFIERS:
Polar: logreg/70pct
Sentiment: logreg/60pct
Test accuracy: 0.649
Precision for neg: 0.667
Recall for neg: 0.701
F measure for neg: 0.683
Precision for neu: 0.654
Recall for neu: 0.504
F measure for neu: 0.569
Precision for pos: 0.629
Recall for pos: 0.709
F measure for pos: 0.667



CLASSIFIERS:
Polar: logreg/70pct
Sentiment: logreg/70pct
Test accuracy: 0.643
Precision for neg: 0.654
Recall for neg: 0.706
F measure for neg: 0.679
Precision for neu: 0.654
Recall for neu: 0.504
F measure for neu: 0.569
Precision for pos: 0.625
Recall for pos: 0.687
F measure for pos: 0.654



CLASSIFIERS:
Polar: logreg/70pct
Sentiment: logreg/80pct
Test accuracy: 0.645
Precision for neg: 0.645
Recall for neg: 0.718
F measure for neg: 0.679
Precision for neu: 0.654
Recall for neu: 0.504
F measure for neu: 0.569
Precision for pos: 0.639
Recall for pos: 0.681
F measure for pos: 0.660



CLASSIFIERS:
Polar: logreg/80pct
Sentiment: multiNB/60pct
Test accuracy: 0.618
Precision for neg: 0.758
Recall for neg: 0.531
F measure for neg: 0.625
Precision for neu: 0.605
Recall for neu: 0.496
F measure for neu: 0.545
Precision for pos: 0.558
Recall for pos: 0.797
F measure for pos: 0.656



CLASSIFIERS:
Polar: logreg/80pct
Sentiment: multiNB/70pct
Test accuracy: 0.622
Precision for neg: 0.744
Recall for neg: 0.559
F measure for neg: 0.639
Precision for neu: 0.605
Recall for neu: 0.496
F measure for neu: 0.545
Precision for pos: 0.566
Recall for pos: 0.780
F measure for pos: 0.656



CLASSIFIERS:
Polar: logreg/80pct
Sentiment: multiNB/80pct
Test accuracy: 0.633
Precision for neg: 0.741
Recall for neg: 0.599
F measure for neg: 0.662
Precision for neu: 0.605
Recall for neu: 0.496
F measure for neu: 0.545
Precision for pos: 0.581
Recall for pos: 0.769
F measure for pos: 0.662



CLASSIFIERS:
Polar: logreg/80pct
Sentiment: logreg/60pct
Test accuracy: 0.641
Precision for neg: 0.659
Recall for neg: 0.689
F measure for neg: 0.674
Precision for neu: 0.605
Recall for neu: 0.496
F measure for neu: 0.545
Precision for pos: 0.643
Recall for pos: 0.703
F measure for pos: 0.672



CLASSIFIERS:
Polar: logreg/80pct
Sentiment: logreg/70pct
Test accuracy: 0.635
Precision for neg: 0.647
Recall for neg: 0.695
F measure for neg: 0.670
Precision for neu: 0.605
Recall for neu: 0.496
F measure for neu: 0.545
Precision for pos: 0.639
Recall for pos: 0.681
F measure for pos: 0.660



CLASSIFIERS:
Polar: logreg/80pct
Sentiment: logreg/80pct
Test accuracy: 0.637
Precision for neg: 0.638
Recall for neg: 0.706
F measure for neg: 0.670
Precision for neu: 0.605
Recall for neu: 0.496
F measure for neu: 0.545
Precision for pos: 0.654
Recall for pos: 0.676
F measure for pos: 0.665



FEATURES: uni-bi-extra-mpqa-subj
CLASSIFIERS:
Polar: multiNB/60pct
Sentiment: multiNB/60pct
Test accuracy: 0.627
Precision for neg: 0.735
Recall for neg: 0.548
F measure for neg: 0.628
Precision for neu: 0.734
Recall for neu: 0.496
F measure for neu: 0.592
Precision for pos: 0.537
Recall for pos: 0.802
F measure for pos: 0.643



CLASSIFIERS:
Polar: multiNB/60pct
Sentiment: multiNB/70pct
Test accuracy: 0.637
Precision for neg: 0.730
Recall for neg: 0.582
F measure for neg: 0.648
Precision for neu: 0.734
Recall for neu: 0.496
F measure for neu: 0.592
Precision for pos: 0.551
Recall for pos: 0.797
F measure for pos: 0.652



CLASSIFIERS:
Polar: multiNB/60pct
Sentiment: multiNB/80pct
Test accuracy: 0.649
Precision for neg: 0.727
Recall for neg: 0.633
F measure for neg: 0.677
Precision for neu: 0.734
Recall for neu: 0.496
F measure for neu: 0.592
Precision for pos: 0.568
Recall for pos: 0.780
F measure for pos: 0.657



CLASSIFIERS:
Polar: multiNB/60pct
Sentiment: logreg/60pct
Test accuracy: 0.665
Precision for neg: 0.687
Recall for neg: 0.706
F measure for neg: 0.696
Precision for neu: 0.734
Recall for neu: 0.496
F measure for neu: 0.592
Precision for pos: 0.617
Recall for pos: 0.753
F measure for pos: 0.678



CLASSIFIERS:
Polar: multiNB/60pct
Sentiment: logreg/70pct
Test accuracy: 0.665
Precision for neg: 0.674
Recall for neg: 0.712
F measure for neg: 0.692
Precision for neu: 0.734
Recall for neu: 0.496
F measure for neu: 0.592
Precision for pos: 0.627
Recall for pos: 0.747
F measure for pos: 0.682



CLASSIFIERS:
Polar: multiNB/60pct
Sentiment: logreg/80pct
Test accuracy: 0.673
Precision for neg: 0.674
Recall for neg: 0.734
F measure for neg: 0.703
Precision for neu: 0.734
Recall for neu: 0.496
F measure for neu: 0.592
Precision for pos: 0.645
Recall for pos: 0.747
F measure for pos: 0.692



CLASSIFIERS:
Polar: multiNB/70pct
Sentiment: multiNB/60pct
Test accuracy: 0.624
Precision for neg: 0.735
Recall for neg: 0.548
F measure for neg: 0.628
Precision for neu: 0.722
Recall for neu: 0.504
F measure for neu: 0.593
Precision for pos: 0.535
Recall for pos: 0.791
F measure for pos: 0.639



CLASSIFIERS:
Polar: multiNB/70pct
Sentiment: multiNB/70pct
Test accuracy: 0.635
Precision for neg: 0.730
Recall for neg: 0.582
F measure for neg: 0.648
Precision for neu: 0.722
Recall for neu: 0.504
F measure for neu: 0.593
Precision for pos: 0.550
Recall for pos: 0.786
F measure for pos: 0.647



CLASSIFIERS:
Polar: multiNB/70pct
Sentiment: multiNB/80pct
Test accuracy: 0.647
Precision for neg: 0.727
Recall for neg: 0.633
F measure for neg: 0.677
Precision for neu: 0.722
Recall for neu: 0.504
F measure for neu: 0.593
Precision for pos: 0.567
Recall for pos: 0.769
F measure for pos: 0.653



CLASSIFIERS:
Polar: multiNB/70pct
Sentiment: logreg/60pct
Test accuracy: 0.663
Precision for neg: 0.687
Recall for neg: 0.706
F measure for neg: 0.696
Precision for neu: 0.722
Recall for neu: 0.504
F measure for neu: 0.593
Precision for pos: 0.616
Recall for pos: 0.742
F measure for pos: 0.673



CLASSIFIERS:
Polar: multiNB/70pct
Sentiment: logreg/70pct
Test accuracy: 0.663
Precision for neg: 0.674
Recall for neg: 0.712
F measure for neg: 0.692
Precision for neu: 0.722
Recall for neu: 0.504
F measure for neu: 0.593
Precision for pos: 0.626
Recall for pos: 0.736
F measure for pos: 0.677



CLASSIFIERS:
Polar: multiNB/70pct
Sentiment: logreg/80pct
Test accuracy: 0.671
Precision for neg: 0.674
Recall for neg: 0.734
F measure for neg: 0.703
Precision for neu: 0.722
Recall for neu: 0.504
F measure for neu: 0.593
Precision for pos: 0.644
Recall for pos: 0.736
F measure for pos: 0.687



CLASSIFIERS:
Polar: multiNB/80pct
Sentiment: multiNB/60pct
Test accuracy: 0.627
Precision for neg: 0.735
Recall for neg: 0.548
F measure for neg: 0.628
Precision for neu: 0.717
Recall for neu: 0.511
F measure for neu: 0.597
Precision for pos: 0.539
Recall for pos: 0.791
F measure for pos: 0.641



CLASSIFIERS:
Polar: multiNB/80pct
Sentiment: multiNB/70pct
Test accuracy: 0.635
Precision for neg: 0.729
Recall for neg: 0.576
F measure for neg: 0.644
Precision for neu: 0.717
Recall for neu: 0.511
F measure for neu: 0.597
Precision for pos: 0.552
Recall for pos: 0.786
F measure for pos: 0.649



CLASSIFIERS:
Polar: multiNB/80pct
Sentiment: multiNB/80pct
Test accuracy: 0.647
Precision for neg: 0.725
Recall for neg: 0.627
F measure for neg: 0.673
Precision for neu: 0.717
Recall for neu: 0.511
F measure for neu: 0.597
Precision for pos: 0.569
Recall for pos: 0.769
F measure for pos: 0.654



CLASSIFIERS:
Polar: multiNB/80pct
Sentiment: logreg/60pct
Test accuracy: 0.663
Precision for neg: 0.685
Recall for neg: 0.701
F measure for neg: 0.693
Precision for neu: 0.717
Recall for neu: 0.511
F measure for neu: 0.597
Precision for pos: 0.619
Recall for pos: 0.742
F measure for pos: 0.675



CLASSIFIERS:
Polar: multiNB/80pct
Sentiment: logreg/70pct
Test accuracy: 0.663
Precision for neg: 0.672
Recall for neg: 0.706
F measure for neg: 0.689
Precision for neu: 0.717
Recall for neu: 0.511
F measure for neu: 0.597
Precision for pos: 0.629
Recall for pos: 0.736
F measure for pos: 0.678



CLASSIFIERS:
Polar: multiNB/80pct
Sentiment: logreg/80pct
Test accuracy: 0.671
Precision for neg: 0.672
Recall for neg: 0.729
F measure for neg: 0.699
Precision for neu: 0.717
Recall for neu: 0.511
F measure for neu: 0.597
Precision for pos: 0.647
Recall for pos: 0.736
F measure for pos: 0.689



CLASSIFIERS:
Polar: logreg/60pct
Sentiment: multiNB/60pct
Test accuracy: 0.633
Precision for neg: 0.756
Recall for neg: 0.542
F measure for neg: 0.632
Precision for neu: 0.787
Recall for neu: 0.453
F measure for neu: 0.575
Precision for pos: 0.536
Recall for pos: 0.857
F measure for pos: 0.660



CLASSIFIERS:
Polar: logreg/60pct
Sentiment: multiNB/70pct
Test accuracy: 0.643
Precision for neg: 0.750
Recall for neg: 0.576
F measure for neg: 0.652
Precision for neu: 0.787
Recall for neu: 0.453
F measure for neu: 0.575
Precision for pos: 0.550
Recall for pos: 0.852
F measure for pos: 0.668



CLASSIFIERS:
Polar: logreg/60pct
Sentiment: multiNB/80pct
Test accuracy: 0.653
Precision for neg: 0.743
Recall for neg: 0.621
F measure for neg: 0.677
Precision for neu: 0.787
Recall for neu: 0.453
F measure for neu: 0.575
Precision for pos: 0.563
Recall for pos: 0.835
F measure for pos: 0.673



CLASSIFIERS:
Polar: logreg/60pct
Sentiment: logreg/60pct
Test accuracy: 0.669
Precision for neg: 0.694
Recall for neg: 0.706
F measure for neg: 0.700
Precision for neu: 0.787
Recall for neu: 0.453
F measure for neu: 0.575
Precision for pos: 0.609
Recall for pos: 0.797
F measure for pos: 0.690



CLASSIFIERS:
Polar: logreg/60pct
Sentiment: logreg/70pct
Test accuracy: 0.669
Precision for neg: 0.685
Recall for neg: 0.712
F measure for neg: 0.698
Precision for neu: 0.787
Recall for neu: 0.453
F measure for neu: 0.575
Precision for pos: 0.615
Recall for pos: 0.791
F measure for pos: 0.692



CLASSIFIERS:
Polar: logreg/60pct
Sentiment: logreg/80pct
Test accuracy: 0.677
Precision for neg: 0.677
Recall for neg: 0.734
F measure for neg: 0.705
Precision for neu: 0.787
Recall for neu: 0.453
F measure for neu: 0.575
Precision for pos: 0.637
Recall for pos: 0.791
F measure for pos: 0.706



CLASSIFIERS:
Polar: logreg/70pct
Sentiment: multiNB/60pct
Test accuracy: 0.639
Precision for neg: 0.754
Recall for neg: 0.537
F measure for neg: 0.627
Precision for neu: 0.761
Recall for neu: 0.482
F measure for neu: 0.590
Precision for pos: 0.549
Recall for pos: 0.857
F measure for pos: 0.670



CLASSIFIERS:
Polar: logreg/70pct
Sentiment: multiNB/70pct
Test accuracy: 0.649
Precision for neg: 0.754
Recall for neg: 0.571
F measure for neg: 0.650
Precision for neu: 0.761
Recall for neu: 0.482
F measure for neu: 0.590
Precision for pos: 0.562
Recall for pos: 0.852
F measure for pos: 0.677



CLASSIFIERS:
Polar: logreg/70pct
Sentiment: multiNB/80pct
Test accuracy: 0.659
Precision for neg: 0.747
Recall for neg: 0.616
F measure for neg: 0.675
Precision for neu: 0.761
Recall for neu: 0.482
F measure for neu: 0.590
Precision for pos: 0.576
Recall for pos: 0.835
F measure for pos: 0.682



CLASSIFIERS:
Polar: logreg/70pct
Sentiment: logreg/60pct
Test accuracy: 0.675
Precision for neg: 0.701
Recall for neg: 0.701
F measure for neg: 0.701
Precision for neu: 0.761
Recall for neu: 0.482
F measure for neu: 0.590
Precision for pos: 0.622
Recall for pos: 0.797
F measure for pos: 0.699



CLASSIFIERS:
Polar: logreg/70pct
Sentiment: logreg/70pct
Test accuracy: 0.675
Precision for neg: 0.691
Recall for neg: 0.706
F measure for neg: 0.698
Precision for neu: 0.761
Recall for neu: 0.482
F measure for neu: 0.590
Precision for pos: 0.629
Recall for pos: 0.791
F measure for pos: 0.701



CLASSIFIERS:
Polar: logreg/70pct
Sentiment: logreg/80pct
Test accuracy: 0.683
Precision for neg: 0.686
Recall for neg: 0.729
F measure for neg: 0.707
Precision for neu: 0.761
Recall for neu: 0.482
F measure for neu: 0.590
Precision for pos: 0.649
Recall for pos: 0.791
F measure for pos: 0.713



CLASSIFIERS:
Polar: logreg/80pct
Sentiment: multiNB/60pct
Test accuracy: 0.633
Precision for neg: 0.754
Recall for neg: 0.537
F measure for neg: 0.627
Precision for neu: 0.720
Recall for neu: 0.482
F measure for neu: 0.578
Precision for pos: 0.548
Recall for pos: 0.841
F measure for pos: 0.664



CLASSIFIERS:
Polar: logreg/80pct
Sentiment: multiNB/70pct
Test accuracy: 0.643
Precision for neg: 0.754
Recall for neg: 0.571
F measure for neg: 0.650
Precision for neu: 0.720
Recall for neu: 0.482
F measure for neu: 0.578
Precision for pos: 0.561
Recall for pos: 0.835
F measure for pos: 0.671



CLASSIFIERS:
Polar: logreg/80pct
Sentiment: multiNB/80pct
Test accuracy: 0.653
Precision for neg: 0.747
Recall for neg: 0.616
F measure for neg: 0.675
Precision for neu: 0.720
Recall for neu: 0.482
F measure for neu: 0.578
Precision for pos: 0.575
Recall for pos: 0.819
F measure for pos: 0.676



CLASSIFIERS:
Polar: logreg/80pct
Sentiment: logreg/60pct
Test accuracy: 0.665
Precision for neg: 0.697
Recall for neg: 0.689
F measure for neg: 0.693
Precision for neu: 0.720
Recall for neu: 0.482
F measure for neu: 0.578
Precision for pos: 0.617
Recall for pos: 0.780
F measure for pos: 0.689



CLASSIFIERS:
Polar: logreg/80pct
Sentiment: logreg/70pct
Test accuracy: 0.665
Precision for neg: 0.687
Recall for neg: 0.695
F measure for neg: 0.691
Precision for neu: 0.720
Recall for neu: 0.482
F measure for neu: 0.578
Precision for pos: 0.624
Recall for pos: 0.775
F measure for pos: 0.691



CLASSIFIERS:
Polar: logreg/80pct
Sentiment: logreg/80pct
Test accuracy: 0.673
Precision for neg: 0.683
Recall for neg: 0.718
F measure for neg: 0.700
Precision for neu: 0.720
Recall for neu: 0.482
F measure for neu: 0.578
Precision for pos: 0.644
Recall for pos: 0.775
F measure for pos: 0.703



"""