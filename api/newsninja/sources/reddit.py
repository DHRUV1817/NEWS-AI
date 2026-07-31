"""Reddit via the official OAuth API.

The public search.json endpoint returns 403 (verified 2026-07-31), so a
script app at reddit.com/prefs/apps is required. Without credentials this
source reports itself unavailable rather than failing at call time.
"""

from collections.abc import Callable
from typing import Any

from newsninja.errors import SourceError
from newsninja.models import Article


def _default_client_factory(**kwargs: Any) -> Any:
    import praw

    return praw.Reddit(**kwargs)


class RedditSource:
    name = "reddit"

    def __init__(
        self,
        client_id: str | None,
        client_secret: str | None,
        user_agent: str,
        client_factory: Callable[..., Any] = _default_client_factory,
    ) -> None:
        self._client_id = client_id
        self._client_secret = client_secret
        self._user_agent = user_agent
        self._factory = client_factory

    def available(self) -> bool:
        return bool(self._client_id and self._client_secret)

    def fetch(self, topic: str, limit: int = 8) -> list[Article]:
        if not self.available():
            raise SourceError(
                self.name,
                "credentials missing; create a script app at reddit.com/prefs/apps "
                "and set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET",
            )

        try:
            reddit = self._factory(
                client_id=self._client_id,
                client_secret=self._client_secret,
                user_agent=self._user_agent,
            )
            submissions = list(
                reddit.subreddit("all").search(
                    topic, sort="hot", time_filter="week", limit=limit
                )
            )
        except Exception as exc:
            raise SourceError(self.name, str(exc)) from exc

        articles: list[Article] = []
        for submission in submissions:
            body = (
                f"{submission.title}. Reddit discussion with {submission.score} "
                f"upvotes and {submission.num_comments} comments. {submission.selftext}"
            ).strip()
            articles.append(
                Article(
                    title=submission.title,
                    url=f"https://www.reddit.com{submission.permalink}",
                    source=self.name,
                    body=body,
                )
            )
        return articles
