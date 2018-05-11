from django.test import TestCase
from django.test import Client


class LstmTestCase(TestCase):
    def setUp(self):
        self.client = Client(enforce_csrf_checks=True)

    def tearDown(self):
        pass

    def test_the_home_page(self):
        response = self.client.get("/")
        self.assertTemplateUsed(response, "base.html")
        self.assertTemplateUsed(response, "main/home_page.html")

    def test_lstm_page(self):
        response = self.client.get("/try-lstm")
        self.assertTemplateUsed(response, "base.html")
        self.assertTemplateUsed(response, "main/try_lstm.html")

    def test_the_ajax_request_and_response(self):
        r = self.client.post('/try-lstm',
                             {'origin_message': 'hello',
                                 'response_message': 'hello'},
                             **{'HTTP_X_REQUESTED_WITH': 'XMLHttpRequest'})
        print(r)
