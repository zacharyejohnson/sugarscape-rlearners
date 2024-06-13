class Patch:
    """
    Represents a patch in the Sugarscape model. Each patch holds a quantity of a resource (good) and can be occupied by an agent.
    """

    def __init__(self, model, row, col, maxQ, good):
        """
        Initializes a new Patch instance.

        Args:
            model: Reference to the model instance that this patch belongs to.
            row (int): The row position of the patch in the grid.
            col (int): The column position of the patch in the grid.
            maxQ (int): The maximum quantity of the resource that this patch can hold.
            good (str): The type of resource (good) that this patch contains.

        Attributes:
            Q (int): The current quantity of the resource available in this patch.
            agent: The agent currently occupying this patch, if any.
        """
        self.model = model  # Link to the model instance
        self.row = row      # Row position in the grid
        self.col = col      # Column position in the grid
        self.maxQ = maxQ    # Maximum quantity of the resource
        self.Q = maxQ       # Current quantity of the resource
        self.good = good    # Type of resource
        self.agent = None   # Agent occupying the patch (None if unoccupied)
