from typing import Any, Callable, Dict, List


class Element:
    def __init__(self, function: Callable, name: str) -> None:
        self.function = function
        self.name = name
        self.headers = {}
        self.run_settings = {}

    def __call__(self, image):
        result = self.function(image)
        self.headers = {}
        self.run_settings = {}
        return result

    def __str__(self):
        return self.name

    def set_headers(self, headers: Dict[str, Any]):
        self.headers = headers

    def set_run_settings(self, run_settings: Dict[str, Any]):
        if run_settings is not None:
            self.run_settings = run_settings


class Pipeline(Element):
    def __init__(self, elements: List[Element]) -> None:
        self.elements = elements
        super().__init__(self.apply, "Pipeline")

    def apply(self, image, run_settings: Dict[str, Any] = None, headers: Dict[str, Any] = None):
        if headers is None:
            headers = {}
        for element in self.elements:
            element.set_headers(headers)
            element.set_run_settings(run_settings)
            image = element(image)
        return image, headers
