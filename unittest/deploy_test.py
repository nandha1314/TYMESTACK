import unittest
from unittest.mock import patch
import os
import shutil

class TestModelDeployment(unittest.TestCase):

    @patch('os.path.exists')
    @patch('os.remove')
    @patch('shutil.move')
    def test_deploy_new_model_when_previous_exists(self, mock_move, mock_remove, mock_exists):
        # Set up mock responses for file existence
        mock_exists.side_effect = lambda path: path == 'models/new_model.joblib' or path == 'models/previous_model.joblib'

        # Call the deployment code
        new_model_path = 'models/new_model.joblib'
        previous_model_path = 'models/previous_model.joblib'

        if os.path.exists(new_model_path):
            if os.path.exists(previous_model_path):
                os.remove(previous_model_path)
            shutil.move(new_model_path, previous_model_path)

        # Check that previous model was removed and new model moved
        mock_remove.assert_called_once_with(previous_model_path)
        mock_move.assert_called_once_with(new_model_path, previous_model_path)

    @patch('os.path.exists')
    @patch('shutil.move')
    def test_deploy_new_model_when_no_previous_model(self, mock_move, mock_exists):
        # Set up mock responses to simulate no previous model
        mock_exists.side_effect = lambda path: path == 'models/new_model.joblib'

        # Call the deployment code
        new_model_path = 'models/new_model.joblib'
        previous_model_path = 'models/previous_model.joblib'

        if os.path.exists(new_model_path):
            if os.path.exists(previous_model_path):
                os.remove(previous_model_path)
            shutil.move(new_model_path, previous_model_path)

        # Check that new model was moved without removing a previous model
        mock_move.assert_called_once_with(new_model_path, previous_model_path)

    @patch('os.path.exists')
    def test_no_new_model_to_deploy(self, mock_exists):
        # Set up mock response to simulate no new model
        mock_exists.return_value = False

        # Call the deployment code
        new_model_path = 'models/new_model.joblib'
        previous_model_path = 'models/previous_model.joblib'

        if os.path.exists(new_model_path):
            if os.path.exists(previous_model_path):
                os.remove(previous_model_path)
            shutil.move(new_model_path, previous_model_path)

        # Check that no move or remove was called
        mock_exists.assert_called_once_with(new_model_path)

if __name__ == '__main__':
    unittest.main()
