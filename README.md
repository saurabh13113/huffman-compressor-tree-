# huffman-file-compressor 🌲
University project that decompresses file sizes using Huffman trees.

The core idea behind Huffman coding is to assign shorter codes to more frequently occurring characters and longer codes to less frequent ones, thereby reducing the overall file size. Here's a step-by-step breakdown of how my file system compressor works:

Frequency Analysis: First, I read the entire file and count the frequency of each character. This helps in understanding which characters appear more often and which are rarer.

Building the Huffman Tree: With the frequency data, I construct a priority queue where each character is a node. Nodes with lower frequencies are given higher priority. By repeatedly combining the two least frequent nodes into a new node (whose frequency is the sum of the two), I build a binary tree called the Huffman tree. The root of this tree represents the entire dataset.

Generating Huffman Codes: Once the tree is built, I traverse it to generate the Huffman codes. Each left traversal represents a binary '0', and each right traversal represents a binary '1'. By following this path from the root to each leaf (character), I assign a unique binary code to each character, with more frequent characters having shorter codes.

Encoding the File: Using the generated Huffman codes, I encode the original file. Each character in the file is replaced with its corresponding Huffman code, transforming the file into a binary string.

Writing Compressed Data: The encoded binary string, along with a header containing the Huffman tree structure (to allow for decoding later), is written to a new compressed file.

Decompression: To decompress the file, I read the Huffman tree structure from the header, then use it to decode the binary string back into the original characters by traversing the tree.

By using Huffman trees for compression, I achieve significant file size reduction, especially with files containing many repetitive characters. This method ensures that no data is lost, providing a perfect balance between efficiency and accuracy. This project not only deepened my understanding of data compression but also showcased the practical applications of algorithms in real-world scenarios.

CITATIONS:
1.https://www.csfieldguide.org.nz/en/interactives/huffman-tree/
 To better understand HuffmanTrees and visualize them and confirm that the Huffman-trees being built are right.
2.https://www.youtube.com/watch?v=0kNXhFIEd_w
 Video explaining on Huffman coding and Huffman trees to understand how Huffman trees are made and why they are efficient.
3.https://cmps-people.ok.ubc.ca/ylucet/DS/Huffman.html
 Huffman-tree visualizer giving realtime animation on how a huffman tree is being built, useful for debugging and code analysis.
