
text1= "Neo-Nazism consists of post-World War II militant social or political movements seeking to revive and implement the ideology of Nazism. Neo-Nazis seek to employ their ideology to promote hatred and attack minorities, or in some cases to create a fascist political state. It is a global phenomenon, with organized representation in many countries and international networks. It borrows elements from Nazi doctrine, including ultranationalism, racism, xenophobia, ableism, homophobia, anti-Romanyism, antisemitism, anti-communism and initiating the Fourth Reich. Holocaust denial is a common feature, as is the incorporation of Nazi symbols and admiration of Adolf Hitler. In some European and Latin American countries, laws prohibit the expression of pro-Nazi, racist, anti-Semitic, or homophobic views. Many Nazi-related symbols are banned in European countries (especially Germany) in an effort to curtail neo-Nazism. The term neo-Nazism describes any post-World War II militant, social or political movements seeking to revive the ideology of Nazism in whole or in part. The term neo-Nazism can also refer to the ideology of these movements, which may borrow elements from Nazi doctrine, including ultranationalism, anti-communism, racism, ableism, xenophobia, homophobia, anti-Romanyism, antisemitism, up to initiating the Fourth Reich. Holocaust denial is a common feature, as is the incorporation of Nazi symbols and admiration of Adolf Hitler. Neo-Nazism is considered a particular form of far-right politics and right-wing extremism."

text2 = "In our modern world, there are many factors that place the wellbeing of the planet in jeopardy. While some people have the opinion that environmental problems are just a natural occurrence, others believe that human beings have a huge impact on the environment. Regardless of your viewpoint, take into consideration the following factors that place our environment as well as the planet Earth danger.Global warming or climate change is a major contributing factor to environmental damage. Because of global warming, we have seen an increase in melting ice caps, a rise in sea levels, and the formation of new weather patterns. These weather patterns have caused stronger storms, droughts, and flooding in places that they formerly did not occur.Air pollution is primarily caused as a result of excessive and unregulated emissions of carbon dioxide into the air. Pollutants mostly emerge from the burning of fossil fuels in addition to chemicals, toxic substances, and improper waste disposal. Air pollutants are absorbed into the atmosphere, and they can cause smog, a combination of smoke and fog, in valleys as well as produce acidic precipitation in areas far away from the pollution source.In many areas, people and local governments do not sustainably use their natural resources. Mining for natural gases, deforestation, and even improper use of water resources can have tremendous effects on the environment. While these strategies often attempt to boost local economies, their effects can lead to oil spills, interrupted animal habitats, and droughts.Ultimately, the effects of the modern world on the environment can lead to many problems. Human beings need to consider the repercussions of their actions, trying to reduce, reuse, and recycle materials while establishing environmentally sustainable habits. If measures are not taken to protect the environment, we can potentially witness the extinction of more endangered species, worldwide pollution, and a completely uninhabitable planet."

text3 = "Valentine's Day (or Saint Valentine's Day) is a holiday that, in the United States, takes place on February 14, and technically signifies the accomplishments of St. Valentine, a third-century Roman saint.With that said, most Americans, instead of honoring St. Valentine through religious ceremony, enjoy the holiday by engaging in romantic behavior with their significant other or someone who they wish to be their significant other; gifts, special dinners, and other acknowledgements of affection comprise most individuals' Valentine's Day celebrations.Chocolates and flowers are commonly given as gifts during Valentine's Day, as are accompanying greeting cards (greeting card companies release new Valentine's Day designs annually). Red and pink are generally understood to be the colors of Valentine's Day, and many individuals, instead of celebrating romantically, spend the holiday with their friends and/or family members.Variations of Valentine's Day are celebrated across the globe throughout the year. In America, the holiday, although acknowledged by the vast majority of the population, isn't federally recognized; no time off work is granted for Valentine's Day."

# TEXT 1
from summa.summarizer import summarize
# Define length of the summary as a proportion of the text
print(summarize(text1, ratio=0.2))
print("\n")
print(summarize(text1, words=30))
print("\n")

from summa import keywords 
print("Keywords:\n",keywords.keywords(text1))
print("\n")
# to print the top 3 keywords
print("Top 3 Keywords:\n",keywords.keywords(text1,words=3))

print("\n")

# TEXT 2
from summa.summarizer import summarize
# Define length of the summary as a proportion of the text
print(summarize(text2, ratio=0.2))
print("\n")
print(summarize(text2, words=30))
print("\n")

from summa import keywords 
print("Keywords:\n",keywords.keywords(text2))
print("\n")
# to print the top 3 keywords
print("Top 3 Keywords:\n",keywords.keywords(text2,words=3))

print("\n")

# TEXT 3
from summa.summarizer import summarize
# Define length of the summary as a proportion of the text
print(summarize(text3, ratio=0.4))
print("\n")
print(summarize(text3, words=52))

from summa import keywords 
print("Keywords:\n",keywords.keywords(text3))
print("\n")
# to print the top 3 keywords
print("Top 3 Keywords:\n",keywords.keywords(text3,words=3))