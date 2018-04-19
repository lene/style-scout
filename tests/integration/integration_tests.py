"""Temporary parking space for tests that need to be run as integration tests."""


class EbayDownloaderIOTest:
    def test_load_weights_from_saved_weights_equal_original_weights(self):
        assert False, 'Test not yet implemented - should be integration test'


class TrainingRunnerTest:
    def test_different_optimizers(self):
        assert False, "Test not yet implemented - should be integration test"

    def test_different_fully_connected_layers(self):
        assert False, "Test not yet implemented - should be integration test"
        for nn_type in ('inception', 'xception', 'vgg16', 'vgg19', 'resnet50'):
            args = Args.default_args()
            args.type = nn_type
            layers = TrainingRunner(args).instantiate_model(None).layers
            args.layers = [1024, 1024]
            # this instantiates a full session. better solution needed.
            self.assertEqual(
                len(TrainingRunner(args).instantiate_model(None).layers), len(layers) + 1
            )

