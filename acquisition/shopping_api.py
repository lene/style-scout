from typing import Any, Dict, List


class ShoppingApi:
    def categories(self, root_id: int) -> List[Any]:
        raise NotImplementedError()

    def get_category_items(self, category: Any, limit: int=100, page: int=1) -> Any:
        raise NotImplementedError()

    def get_item(self, item_id: int) -> Dict:
        raise NotImplementedError()
