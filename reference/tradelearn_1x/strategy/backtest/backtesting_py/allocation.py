import functools
from numbers import Number

import pandas as pd


class Allocation:
    '''The `Allocation` class manages the allocation of values among different assets in a portfolio. It provides
    methods for creating and managing asset buckets, assigning weights to assets, and merging the weights into the
    parent allocation object.

    `Allocation` is not meant to be instantiated directly. Instead, it is created automatically when a new
    `Strategy` object is created. The `Allocation` object is accessed through the `Strategy.alloc` property.

    The `Allocation` object is used as an input to the `Strategy.rebalance()` method to rebalance the portfolio according
    to the current weight allocation.

    `Allocation` has the following notable properties:

    - `tickers`: A list of tickers representing the asset space in the allocation.
    - `weights`: The weight allocation to be used in the current rebalance cycle.
    - `previous_weights`: The weight allocation used in the previous rebalance cycle.
    - `unallocated`: Unallocated equity weight, i.e. 1 minus the sum of weights already allocated to assets.
    - `bucket`: Bucket accessor for weight allocation.

    `Allocation` provides two ways to assign weights to assets:

    1. Explicitly assign weights to assets using `Allocation.weights` property.

        It's possible to assign weights to individual asset or to all assets in the asset space as a whole. Not all weights
        need to be specified. If an asset is not assigned a weight, it will have a weight of 0.

        Example:
        ```python
        # Assign weight to individual asset
        strategy.alloc.weights['A'] = 0.5

        # Assign weight to all assets
        strategy.alloc.weights = pd.Series([0.1, 0.2, 0.3], index=['A', 'B', 'C'])
        ```

    2. Use `Bucket` to assign weights to logical groups of assets, then merge the weights into the parent allocation object.

        A `Bucket` is a container that groups assets together and provieds methods for weight allocation. Assets can be added
        to the bucket by appending lists or filtering conditions. Weights can be assigned to the assets in the bucket using
        different allocation methods. Multiple buckets can be created for different groups of assets. Once the weight
        allocation is done at bucket level , the weights of the buckets can be merged into those of the parent allocation object.

        Example:
        ```python
        # Create a bucket and add assets to it
        bucket = strategy.alloc.bucket['bucket1']
        bucket.append(['A', 'B', 'C'])

        # Assign weights to the assets in the bucket
        bucket.weight_explicitly([0.1, 0.2, 0.3])

        # Merge the bucket into the parent allocation object
        bucket.apply('update')
        ```

    The state of the `Allocation` object is managed by the `Strategy` object across rebalance cycles. A rebalance
    cycle involves:

    1. Initializing the weight allocation at the beginning of the cycle by calling either `Allocation.assume_zero()`
    to reset all weights to zero or `Allocation.assume_previous()` to inherit the weights from the previous cycle. This
    must be done before any weight allocation attempts.
    2. Adjusting the weight allocation using either explicitly assignment or `Bucket` method.
    3. Calling `Strategy.rebalance()` to rebalance the portfolio according to the current allocation plan.

    After each rebalance cycle, the weight allocation is reset, and the process starts over. At any point, the weight
    allocation from the previous cycle can be accessed using the `previous_weights` property.

    A rebalance cycle is not necessarily equal to the simulation time step. For example, simulation can be done at
    daily frequency, while the portfolio is rebalanced every month. In this case, the weight allocation is maintained
    across multiple time steps until the next time `Strategy.rebalance()` is called.

    Example:
    ```python
    class MyStrategy(Strategy):
        def init(self):
            pass

        def next(self):
            # Initialize the weight allocation
            self.alloc.assume_zero()

            # Adjust the weight allocation
            self.alloc.bucket['equity'].append(['A', 'B', 'C']).weight_equally(sum_=0.4).apply('update')
            self.alloc.bucket['bond'].append(['D', 'E']).weight_equally(sum=_0.4).apply('update')
            self.alloc.weights['gold'] = self.alloc.unallocated

            # Rebalance the portfolio
            self.rebalance()
    ```
    '''

    class Bucket:
        '''`Bucket` is a container that groups assets together and applies weight allocation among them.
        A bucket is associated with a parent allocation object, while the allocation object can be
        associated with multiple buckets.

        Assets in a bucket are identified by their tickers. They are unique within the bucket, but can be
        repeated in different buckets.

        Using `Bucket` for weight allocation takes 3 steps:

        1. Assets are added to the bucket by appending lists or filtering conditions. The rank of the assets
        in the bucket is preserved and can be used to assign weights.
        2. Weights are assigned to the assets using different allocation methods.
        3. Once the weight allocation at bucket level is done, the weights of the bucket can be merged into
        those of the parent allocation object.
        '''

        def __init__(self, alloc: 'Allocation') -> None:
            self._alloc = alloc
            self._tickers = []
            self._weights = None

        @property
        def tickers(self) -> list:
            '''Assets in the bucket. This is a read-only property.'''
            return self._tickers.copy()

        @property
        def weights(self) -> pd.Series:
            '''Weights of the assets in the bucket. This is only available after weight allocation is done
            by calling `Bucket.weight_*()` methods. This is a read-only property.'''
            assert (self._weights >= 0).all(), 'Weight should be non-negative.'
            assert self._weights.sum(
            ) < 1.000000000000001, f'Total weight should be less than or equal to 1. Got {self._weights.sum()}'
            return self._weights.copy()

        def append(self, ranked_list: list | pd.Series, *conditions: list | pd.Series) -> 'Allocation.Bucket':
            '''Add assets that are in the ranked list to the end of the bucket.

            `ranked_list` can be specified in three ways:

            1. A list of assets or anything list-like, all items will be added.
            2. A boolean Series with assets as the index and a True value to indicate the asset should be added.
            3. A non-boolean Series with assets as the index and all assets in the index will be added.

            The rank of the assets is determined by its order in the list or in the index. The rank of the assets
            in the bucket is preserved. If an asset is already in the bucket, its rank in bucket will not be affected
            by appending new list to the bucket, even if the asset is ranked differently in the new list.

            Multiple conditions can be specified as filters to exclude certain assets in the ranked list from being
            added. Assets must satisfy all the conditions in order to be added to the bucket.

            `conditions` can be specified in the same way as `ranked_list`, only that the asset order in a condition
            is not important.

            Example:
            ```python
            # Append 'A' and 'B' to the bucket
            bucket.append(['A', 'B'])

            # Append 'A' and 'C' to the bucket
            bucket.append(pd.Series([True, False, True], index=['A', 'B', 'C']))

            # Append 'C' to the bucket
            bucket.append(pd.Series([1, 2, 3], index=['A', 'B', 'C']).nlargest(2), pd.Series([1, 2, 3], index=['A', 'B', 'C']) > 2)
            ```

            Args:
                ranked_list: A list of assets or a Series of assets to be added to the bucket.
                conditions: A list of assets or a Series of assets to be used as conditions to filter the assets.
            '''
            list_and_conditions = [ranked_list] + list(conditions)
            candidates = {}
            for item in list_and_conditions:
                item = [index for index, value in item.items() if not isinstance(
                    value, bool) or value] if isinstance(item, pd.Series) else list(item)
                for x in item:
                    candidates[x] = candidates.get(x, 0) + 1
            candidates = [x for x in candidates if candidates[x] == len(list_and_conditions)]
            self._tickers.extend([x for x in candidates if x not in self._tickers])
            return self

        def remove(self, *conditions: list | pd.Series) -> 'Allocation.Bucket':
            '''Remove assets that satisify all the given conditions from the bucket.

            `conditions` can be specified in three ways:

            1. A list of assets or anything list-like, all assets will be removed.
            2. A boolean Series with assets as the index and a True value to indicate the asset should be removed.
            3. A non-boolean Series with assets as the index and all assets in the index will be removed.

            Example:
            ```python
            # Remove 'A' and 'B' from the bucket
            bucket.remove(['A', 'B'])

            # Remove 'A' and 'C' from the bucket
            bucket.remove(pd.Series([True, False, True], index=['A', 'B', 'C']))

            # Remove 'A' and 'B' from the bucket
            bucket.remove(pd.Series([1, 2, 3], index=['A', 'B', 'C']).nsmallest(2))

            # Remove 'B' from the bucket
            bucket.remove(pd.Series([1, 2, 3], index=['A', 'B', 'C']) > 1, pd.Series([1, 2, 3], index=['A', 'B', 'C']) < 3)
            ```
            Args:
                conditions: A list of assets or a Series of assets to be used as conditions to filter the assets.
            '''
            if len(conditions) == 0:
                return
            candidates = {}
            for item in conditions:
                item = [index for index, value in item.items() if not isinstance(
                    value, bool) or value] if isinstance(item, pd.Series) else list(item)
                for x in item:
                    candidates[x] = candidates.get(x, 0) + 1
            self._tickers = [x for x in self._tickers if candidates.get(x, 0) < len(conditions)]
            return self

        def trim(self, limit: int) -> 'Allocation.Bucket':
            '''Trim the bucket to a maximum number of assets.

            Args:
                limit: Maximum number of assets should be included
            '''
            self._tickers = self._tickers[:limit]
            return self

        def weight_explicitly(self, weight: float | list | pd.Series) -> 'Allocation.Bucket':
            '''Assign weights to the assets in the bucket.

            `weight` can be specified in three ways:

            1. A single weight should be assigned to all assets in the bucket.
            2. A list of weights should be assigned to the assets in the bucket in rank order. If more weights are provided than the number of assets in the bucket, the extra weights are ignored. If fewer weights are provided, the remaining assets will be assigned a weight of 0.
            3. A Series with assets as the index and the weight as the value. If no weight is provided for an asset, it will be assigned a weight of 0. If a weight is provided for an asset that is not in the bucket, it will be ignored.

            Example:
            ```python
            bucket.append(['A', 'B', 'C']).weight_explicitly(0.2)
            bucket.append(['A', 'B', 'C']).weight_explicitly([0.1, 0.2, 0.3])
            bucket.append(['A', 'B', 'C']).weight_explicitly(pd.Series([0.1, 0.2, 0.3], index=['A', 'B', 'C']))
            ```
            Args:
                weight: A single value, a list of values or a Series of weights.
            '''
            if len(self._tickers) == 0:
                self._weights = pd.Series()
            elif isinstance(weight, Number):
                assert 0 <= weight * len(self._tickers) < 1.000000000000001, 'Total weight should be within [0, 1].'
                self._weights = pd.Series(weight, index=self._tickers)
            elif isinstance(weight, list):
                assert all(0 <= x < 1.000000000000001 for x in weight), 'Weight should be non-negative.'
                assert sum(weight) < 1.000000000000001, 'Total weight should be less than or equal to 1.'
                weight = weight[:len(self._tickers)]
                weight.extend([0.] * (len(self._tickers) - len(weight)))
                self._weights = pd.Series(weight, index=self._tickers)
            elif isinstance(weight, pd.Series):
                assert (weight >= 0).all(), 'Weight should be non-negative.'
                assert weight.sum() < 1.000000000000001, 'Total weight should be less than or equal to 1.'
                weight = weight[weight.index.isin(self._tickers)]
                self._weights = pd.Series(0., index=self._tickers)
                self._weights.loc[weight.index] = weight
            else:
                raise ValueError('Weight should be a single value, a list of values or a Series of weights.')
            return self

        def weight_equally(self, sum_: float = None) -> 'Allocation.Bucket':
            '''Allocate equity value equally to the assets in the bucket.

            `sum_` should be between 0 and 1, with 1 means 100% of value should be allocated.

            Example:
            ```python
            bucket.append(['A', 'B', 'C']).weight_equally(0.5)
            ```

            Args:
                sum_: Total weight that should be allocated.
            '''
            assert sum_ is None or 0 <= sum_ < 1.000000000000001, 'Total weight should be within [0, 1].'
            if sum_ is None:
                sum_ = self._alloc.unallocated
            if len(self._tickers) == 0:
                self._weights = pd.Series()
            else:
                self._weights = pd.Series(1 / len(self._tickers), index=self._tickers) * sum_
            return self

        def weight_proportionally(self, relative_weights: list, sum_: float = None) -> 'Allocation.Bucket':
            '''Allocate equity value proportionally to the assets in the bucket.

            `sum_` should be between 0 and 1, with 1 means 100% of value should be allocated.

            Example:
            ```python
            bucket.append(['A', 'B', 'C']).weight_proportionally([1, 2, 3], 0.5)
            ```

            Args:
                relative_weights: A list of relative weights. The length of the list should be the same as the number of assets in the bucket.
                sum_: Total weight that should be allocated.
            '''
            assert len(relative_weights) == len(
                self._tickers), f'Length of relative_weight {len(relative_weights)} does not match number of assets {len(self._tickers)}'
            assert all(x >= 0 for x in relative_weights), 'Relative weights should be non-negative.'
            assert sum_ is None or 0 <= sum_ < 1.000000000000001, 'Total weight should be within [0, 1].'
            if sum_ is None:
                sum_ = self._alloc.unallocated
            if len(self._tickers) == 0:
                self._weights = pd.Series()
            else:
                self._weights = pd.Series(relative_weights, index=self._tickers) / sum(relative_weights) * sum_
            return self

        def apply(self, method: str = 'update') -> 'Allocation.Bucket':
            '''Apply the weight allocation to the parent allocation object.

            `method` controls how the bucket weight allocation should be merged into the parent allocation object.

            When `method` is `update`, the weights of assets in the bucket will update the weights of the same assets
            in the parent allocation object. If an asset is not in the bucket, its weight in the parent allocation object
            will not be changed. This is the default method.

            When `method` is `overwrite`, the weights of the parent allocation object will be replaced by the weights of the
            assets in the bucket or set to 0 if the asset is not in the bucket.

            When `method` is `accumulate`, the weights of the assets in the bucket will be added to the weights of the same
            assets, while the weights of the assets not in the bucket will remain unchanged.

            If the bucket is empty, no change will be made to the parent allocation object.

            Note that no validation is performed on the weights of the parent allocation object after the bucket weight
            is merged. It is the responsibility of the user to ensure the final weights are valid before use.

            Args:
                method: Method to merge the bucket into the parent allocation object.
                    Available methods are 'update', 'overwrite', 'accumulate'.
            '''
            if self._weights is None:
                raise RuntimeError('Bucket.weight_*() should be called before apply()')
            if self.weights.empty:
                return self
            index = self.weights.index
            if method == 'update':
                self._alloc.weights.loc[index] = self.weights
            elif method == 'overwrite':
                self._alloc.weights.loc[:] = 0.
                self._alloc.weights.loc[index] = self.weights
            elif method == 'accumulate':
                self._alloc.weights.loc[index] = self._alloc.weights.loc[index] + self.weights
            else:
                raise ValueError(f'Invalid method {method}')
            return self

        def __len__(self) -> int:
            return len(self._tickers)

        def __iter__(self):
            return iter(self._tickers)

        def __eq__(self, other):
            if isinstance(other, pd.Series):
                return self._weights.equals(other)
            elif isinstance(other, list):
                return self._tickers == other
            else:
                return False

        def __repr__(self) -> str:
            return f'Bucket(tickers={self._tickers})'

    class BucketGroup:
        def __init__(self, alloc: 'Allocation') -> None:
            self._alloc = alloc
            self._buckets = {}

        def clear(self) -> None:
            self._buckets.clear()

        def __getitem__(self, name: str) -> 'Allocation.Bucket':
            if name not in self._buckets:
                self._buckets[name] = Allocation.Bucket(self._alloc)
            return self._buckets[name]

        def __iter__(self):
            return iter(self._buckets)

        def __len__(self) -> int:
            return len(self._buckets)

    def __init__(self, tickers: list) -> None:
        self._tickers = tickers
        self._previous_weights = pd.Series(0., index=tickers)
        self._weights = None
        self._bucket_group = Allocation.BucketGroup(self)

    def _after_assume(func):
        @functools.wraps(func)
        def inner(self, *args, **kwargs):
            if self._weights is None:
                raise RuntimeError('"Allocation.assume_*()" must be called first.')
            return func(self, *args, **kwargs)
        return inner

    @property
    def tickers(self) -> list:
        '''Assets representing the asset space. This is a read-only property'''
        return self._tickers.copy()

    @property
    @_after_assume
    def bucket(self) -> BucketGroup:
        '''`bucket` provides access to a dictionary of buckets.

        A bucket can be accessed with a string key. If the bucket does not exist, one will be created automatically.

        Buckets are cleared after each rebalance cycle.

        Example:
        ```python
        # Access the bucket named 'equity'
        bucket = strategy.alloc.bucket['equity']
        ```
        '''
        return self._bucket_group

    @property
    @_after_assume
    def weights(self) -> pd.Series:
        '''Current weight allocation. Weight should be non-negative and the total weight should be less than or equal to 1.

        It's possible to assign weights to individual asset or to all assets in the asset space as a whole. When assigning
        weights as a whole, only non-zero weights need to be specified, and other weights are assigned zero automatically.

        Example:
        ```python
        # Assign weight to individual asset
        strategy.alloc.weights['A'] = 0.5

        # Assign weight to all assets
        strategy.alloc.weights = pd.Series([0.1, 0.2, 0.3], index=['A', 'B', 'C'])
        ```
        '''
        assert self._weights.index.to_list() == self._tickers, 'Weight index should be the same as the asset space.'
        assert (self._weights >= 0).all(), 'Weight should be non-negative.'
        assert self._weights.sum() < 1.000000000000001, f'Total weight should be less than or equal to 1. Got {self._weights.sum()}'
        return self._weights

    @weights.setter
    @_after_assume
    def weights(self, value: pd.Series) -> None:
        assert (value >= 0).all(), 'Weight should be non-negative.'
        assert value.sum() < 1.000000000000001, f'Total weight should be less than or equal to 1. Got {value.sum()}'
        self._weights.loc[:] = 0.
        self._weights.loc[value.index] = value

    @property
    def previous_weights(self) -> pd.Series:
        '''Previous weight allocation. This is a read-only property.'''
        return self._previous_weights.copy()

    def assume_zero(self):
        '''Assume all assets have zero weight to begin with in a new rebalance cycle.
        '''
        self._weights = pd.Series(0., index=self.tickers)

    def assume_previous(self):
        '''Assume all assets inherit the same weight as used in the previous rebalance cycle.
        '''
        self._weights = self.previous_weights.copy()

    @property
    @_after_assume
    def unallocated(self) -> float:
        '''Unallocated equity weight. It's the remaining weight that can be allocated to assets. This is a read-only property.'''
        allocated = self._weights.abs().sum()
        assert allocated < 1.000000000000001, f'Total weight should be less than or equal to 1. Got {allocated}'
        return 1. - allocated

    @_after_assume
    def normalize(self):
        '''Normalize the weight allocation so that the sum of weights equals 1.'''
        self._weights = self._weights / self._weights.abs().sum()
        return self.weights

    @property
    @_after_assume
    def modified(self):
        '''True if weight allocation is changed from previous values.'''
        return not self.weights.equals(self.previous_weights)

    def _next(self):
        '''Prepare for the next rebalance cycle. This is called after each call to `Strategy.rebalance()`.
        '''
        self._previous_weights = self._weights.copy()
        self._weights = None
        self._bucket_group.clear()

    def _clear(self):
        '''Clear the weight allocation and buckets.
        '''
        self._previous_weights = pd.Series(0., index=self._tickers)
        self._weights = None
        self._bucket_group.clear()