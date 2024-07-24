WIP

### Load Content



### Chunking

This is perhaps the most crucial component of the whole RAG setup,
we want the splitting to happen at places where it makes sense,
for example the end of a paragraph or a chapter, or where it semantically makes the most sense.
A bad splitting technique would see splitting in the middle of a sentence, or even middle of a word.
An optimised splitting can produce data chunks that have good self-contained information and context,
this would help to improve the end result dramatically.