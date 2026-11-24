# mapclassify

Classification of a single numeric attribute into ordered classes for choropleth mapping.

## Language

### Classes and breaks

**Break**:
The upper bound of a class. Breaks are strictly increasing in a well-formed classification.
_Avoid_: cut, cutpoint, bin edge

**Class**:
An upper-inclusive interval between consecutive breaks: a value belongs to class *j* when it is greater than break *j−1* and at most break *j*. The first class is closed below at the minimum.
_Avoid_: bin, bucket, category

**Empty class**:
A class with positive width that contains no observations. A legitimate outcome of a classifier, not a defect.
_Avoid_: degenerate class

**Zero-width class**:
A class whose lower and upper breaks are equal, so no observation can ever fall in it. Arises only from ties.
_Avoid_: duplicate bin, empty class

**Positional scheme**:
A classification in which each class index has a fixed meaning independent of the data, such as "low outliers" or "one to two standard deviations above the mean". Removing a class would change what the later classes mean.
_Avoid_: fixed classes

### Class counts

**Requested k**:
The number of classes the user asked for.

**k**:
The number of classes a fitted classifier actually produced; always equals the number of breaks and the number of class counts.
_Avoid_: effective k, realized k

### Ties

**Tie**:
Two or more observations with exactly the same value. Tied observations are never split across classes.
_Avoid_: duplicate (when referring to observations)

**Arbitrary choice**:
A point where a classifier must pick among equally good alternatives — equal gaps, equal-cost partitions — for reasons other than tied values. Always resolved deterministically.
_Avoid_: tie (for this meaning), random break
