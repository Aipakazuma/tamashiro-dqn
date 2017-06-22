# tamashiro DQN

"たましろどきゅん"と呼びます。



# Todo

* DQNの実装
* Epsilon greedyの実装



# Learning

## Experience Replay

>学習に使う「状態・アクション・報酬」のセットは、お互いに相関のないものを使ったほうがよいそうです。このためにER(Experience Replay)という手法が重要になります。DQNの最大のポイントの一つのようですね。過去の経験を覚えておいて、そこからランダムで先ほどのセットを取り出したものに対して学習を行う、という手法です。

[Best Experience Replay](http://qiita.com/ashitani/items/bb393e24c20e83e54577#best-experience-replay)



# Reference

* [DQNをKerasとTensorFlowとOpenAI Gymで実装する](https://elix-tech.github.io/ja/2016/06/30/dqn-ja.html)
  * ここのコードをほぼ丸パクリ



# Error List

* TypeError: Population must be a sequence or set.  For dicts, use list(d).
  * [https://stackoverflow.com/questions/40181284/how-to-get-random-sample-from-deque-in-python-3](https://stackoverflow.com/questions/40181284/how-to-get-random-sample-from-deque-in-python-3)
  * `random.sample()`にはlistで渡してね
* ValueError: Sample larger than population
