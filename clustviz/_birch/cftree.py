from pyclustering.cluster import cluster_visualizer
from pyclustering.container.cftree import cftree as cftree_pyclustering
from pyclustering.container.cftree import leaf_node, non_leaf_node, cfnode_type


class cftree(cftree_pyclustering):

    def insert(self, entry):
        """!
        @brief Insert clustering feature to the tree.

        @param[in] entry (cfentry): Clustering feature that should be inserted.

        """
        print("insert entry")
        if self.__root is None:
            print("first time")
            node = leaf_node(entry, None, [entry])

            self.__root = node
            self.__leafes.append(node)

            # Update statistics
            self.__amount_entries += 1
            self.__amount_nodes += 1
            self.__height += 1  # root has successor now
        else:
            print("recursive insert")
            child_node_updation = self.__recursive_insert(entry, self.__root)
            if child_node_updation is True:
                print("try merge_nearest_successors")
                # Splitting has been finished, check for possibility to merge (at least we have already two children).
                if self.__merge_nearest_successors(self.__root) is True:
                    self.__amount_nodes -= 1

    def __recursive_insert(self, entry, search_node):
        """!
        @brief Recursive insert of the entry to the tree.
        @details It performs all required procedures during insertion such as splitting, merging.

        @param[in] entry (cfentry): Clustering feature.
        @param[in] search_node (cfnode): Node from that insertion should be started.

        @return (bool) True if number of nodes at the below level is changed, otherwise False.

        """

        # Non-leaf node
        if search_node.type == cfnode_type.CFNODE_NONLEAF:
            print("insert for non-leaf")
            return self.__insert_for_noneleaf_node(entry, search_node)

        # Leaf is reached
        else:
            print("insert for leaf")
            return self.__insert_for_leaf_node(entry, search_node)

    def __insert_for_leaf_node(self, entry, search_node):
        """!
        @brief Recursive insert entry from leaf node to the tree.

        @param[in] entry (cfentry): Clustering feature.
        @param[in] search_node (cfnode): None-leaf node from that insertion should be started.

        @return (bool) True if number of nodes at the below level is changed, otherwise False.

        """

        node_amount_updation = False

        # Try to absorb by the entity
        index_nearest_entry = search_node.get_nearest_index_entry(
            entry, self.__type_measurement
        )
        nearest_entry = search_node.entries[index_nearest_entry]
        merged_entry = nearest_entry + entry
        print("index_nearest_entry", index_nearest_entry)
        print("nearest entry", nearest_entry)

        print("diam:", merged_entry.get_diameter())
        # Otherwise try to add new entry
        if merged_entry.get_diameter() > self.__threshold:
            print("diam greater than threshold")
            # If it's not exceeded append entity and update feature of the leaf node.
            search_node.insert_entry(entry)

            # Otherwise current node should be splitted

            if len(search_node.entries) > self.__max_entries:
                print("node has to split")
                self.__split_procedure(search_node)
                node_amount_updation = True

            # Update statistics
            self.__amount_entries += 1

        else:
            print("diam ok")
            search_node.entries[index_nearest_entry] = merged_entry
            search_node.feature += entry

        return node_amount_updation

    def __insert_for_noneleaf_node(self, entry, search_node):
        """!
        @brief Recursive insert entry from none-leaf node to the tree.

        @param[in] entry (cfentry): Clustering feature.
        @param[in] search_node (cfnode): None-leaf node from that insertion should be started.

        @return (bool) True if number of nodes at the below level is changed, otherwise False.

        """

        node_amount_updation = False

        min_key = lambda child_node: child_node.get_distance(
            search_node, self.__type_measurement
        )
        nearest_child_node = min(search_node.successors, key=min_key)
        print("nearestchildnode: ", nearest_child_node)
        print("recursive insert in !!!insert_for_nonleaf!!!")
        child_node_updation = self.__recursive_insert(
            entry, nearest_child_node
        )

        # Update clustering feature of none-leaf node.
        search_node.feature += entry

        # Check branch factor, probably some leaf has been splitted and threshold has been exceeded.
        if len(search_node.successors) > self.__branch_factor:
            print("over branch_factor ")

            # Check if it's aleady root then new root should be created (height is increased in this case).
            if search_node is self.__root:
                print("height increases")
                self.__root = non_leaf_node(
                    search_node.feature, None, [search_node]
                )
                search_node.parent = self.__root

                # Update statistics
                self.__amount_nodes += 1
                self.__height += 1

            print("split non-leaf node")
            [new_node1, new_node2] = self.__split_nonleaf_node(search_node)

            # Update parent list of successors
            parent = search_node.parent
            parent.successors.remove(search_node)
            parent.successors.append(new_node1)
            parent.successors.append(new_node2)

            # Update statistics
            self.__amount_nodes += 1
            node_amount_updation = True

        elif child_node_updation is True:
            # Splitting has been finished, check for possibility to merge (at least we have already two children).
            if self.__merge_nearest_successors(search_node) is True:
                self.__amount_nodes -= 1

        return node_amount_updation

    def __merge_nearest_successors(self, node):
        """!
        @brief Find nearest sucessors and merge them.

        @param[in] node (non_leaf_node): Node whose two nearest successors should be merged.

        @return (bool): True if merging has been successfully performed, otherwise False.

        """

        merging_result = False

        if node.successors[0].type == cfnode_type.CFNODE_NONLEAF:
            [
                nearest_child_node1,
                nearest_child_node2,
            ] = node.get_nearest_successors(self.__type_measurement)

            if (
                    len(nearest_child_node1.successors)
                    + len(nearest_child_node2.successors)
                    <= self.__branch_factor
            ):
                node.successors.remove(nearest_child_node2)
                if nearest_child_node2.type == cfnode_type.CFNODE_LEAF:
                    self.__leafes.remove(nearest_child_node2)

                nearest_child_node1.merge(nearest_child_node2)

                merging_result = True

        if merging_result is True:
            print("merging successful")
        else:
            print("merging not successful")
        return merging_result

    def __split_leaf_node(self, node):
        """!
        @brief Performs splitting of the specified leaf node.

        @param[in] node (leaf_node): Leaf node that should be splitted.

        @return (list) New pair of leaf nodes [leaf_node1, leaf_node2].

        @warning Splitted node is transformed to non_leaf.

        """
        print("split leaf")
        # search farthest pair of entries
        [farthest_entity1, farthest_entity2] = node.get_farthest_entries(
            self.__type_measurement
        )
        print("farthest1 ", farthest_entity1)
        print("farthest2 ", farthest_entity2)

        # create new nodes
        new_node1 = leaf_node(
            farthest_entity1, node.parent, [farthest_entity1]
        )
        new_node2 = leaf_node(
            farthest_entity2, node.parent, [farthest_entity2]
        )

        # re-insert other entries
        for entity in node.entries:
            if (entity is not farthest_entity1) and (
                    entity is not farthest_entity2
            ):
                distance1 = new_node1.feature.get_distance(
                    entity, self.__type_measurement
                )
                distance2 = new_node2.feature.get_distance(
                    entity, self.__type_measurement
                )

                if distance1 < distance2:
                    new_node1.insert_entry(entity)
                else:
                    new_node2.insert_entry(entity)

        print("new_node1 ", new_node1)
        print("new_node2 ", new_node2)

        return [new_node1, new_node2]

    def show_feature_distribution(self, data=None):
        """!
         @brief Shows feature distribution.
         @details Only features in 1D, 2D, 3D space can be visualized.

         @param[in] data (list): List of points that will be used for visualization,
         if it not specified than feature will be displayed only.

         """
        visualizer = cluster_visualizer()

        print("amount of nodes: ", self.__amount_nodes)

        if data is not None:
            visualizer.append_cluster(data, marker="x")

        for level in range(0, self.height):
            level_nodes = self.get_level_nodes(level)

            centers = [node.feature.get_centroid() for node in level_nodes]
            visualizer.append_cluster(
                centers, None, markersize=(self.height - level + 1) * 5
            )

        visualizer.show()
