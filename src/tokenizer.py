# src/tokenizer.py

from collections import Counter
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Tokenize:
    def __init__(self, update_queue=None):
        self.update_queue = update_queue
        self.mapping = {}
        self.encode_map = {}
        self.vocab_size = 0

    def bp_encoding(self, text):
        try:
            sorted_set = sorted(set(text))
            self.mapping = {i: ch for i, ch in enumerate(sorted_set)}
            temp = {ch: ch for i, ch in self.mapping.items()}
            idx = len(sorted_set)
            self.len_orig = len(text)
            self.encoded_text = text
            self.max_len = 0

            max_merges = 100  # Limit the number of merges to prevent over-compression
            compression_threshold = 1.2  # Reduced compression ratio threshold

            for i in range(1, max_merges + 1):
                current_length = len(self.encoded_text)
                compression_ratio = self.len_orig / current_length

                # Check termination conditions
                if current_length <= 1 or compression_ratio > compression_threshold:
                    termination_msg = f"BPE completed at iteration {i} with compression ratio {compression_ratio:.2f}"
                    logger.info(termination_msg)
                    if self.update_queue:
                        self.update_queue.put({'type': 'status', 'message': termination_msg})
                    break

                # Count all pairs of consecutive characters
                pairs = Counter(self.encoded_text[j:j + 2] for j in range(len(self.encoded_text) - 1))

                if not pairs:
                    termination_msg = "No more pairs to merge. BPE terminated."
                    logger.info(termination_msg)
                    if self.update_queue:
                        self.update_queue.put({'type': 'status', 'message': termination_msg})
                    break

                # Find the most common pair
                most_common_pair, occurrence = pairs.most_common(1)[0]

                if occurrence == 1:
                    termination_msg = f"Most common pair '{most_common_pair}' occurs only once. BPE terminated."
                    logger.info(termination_msg)
                    if self.update_queue:
                        self.update_queue.put({'type': 'status', 'message': termination_msg})
                    break

                # Merge the most common pair
                new_pair = temp.get(most_common_pair[0], '') + temp.get(most_common_pair[1], '')
                if not new_pair:
                    termination_msg = f"New pair is empty for most common pair '{most_common_pair}'. BPE terminated."
                    logger.info(termination_msg)
                    if self.update_queue:
                        self.update_queue.put({'type': 'status', 'message': termination_msg})
                    break

                self.max_len = max(self.max_len, len(new_pair))
                temp[chr(idx)] = new_pair
                self.mapping[idx] = new_pair
                # Replace all occurrences of the most common pair
                self.encoded_text = self.encoded_text.replace(most_common_pair, chr(idx))
                idx += 1

                # Update vocab size
                self.vocab_size = idx

                # Send status updates every 10 iterations for more frequent feedback
                if i % 10 == 0:
                    status_msg = f"BPE Iteration {i}: Compression ratio {compression_ratio:.2f}"
                    logger.info(status_msg)
                    if self.update_queue:
                        self.update_queue.put({'type': 'status', 'message': status_msg})
                    # Send BPE merge information
                    if self.update_queue:
                        self.update_queue.put({
                            'type': 'bpe_merge',
                            'iteration': i,
                            'merged_pair': most_common_pair,
                            'new_token': chr(idx - 1)  # The new token after merging
                        })

            # Final status message
            final_msg = f"Final Compression Ratio: {self.len_orig / len(self.encoded_text):.2f}"
            logger.info(final_msg)
            if self.update_queue:
                self.update_queue.put({'type': 'status', 'message': final_msg})
            self.encode_map = {ch: i for i, ch in self.mapping.items()}
            logger.info(f"Vocabulary Size after BPE: {self.vocab_size}")

        except Exception as e:
            error_msg = f"Exception during BPE encoding: {str(e)}"
            logger.error(error_msg)
            if self.update_queue:
                self.update_queue.put({'type': 'error', 'message': error_msg})

    def encode(self, text):
        encoded = []
        i = 0
        while i <= (len(text) - self.max_len):
            for j in range(i + self.max_len, i, -1):
                if text[i:j] in self.encode_map:
                    encoded.append(self.encode_map[text[i:j]])
                    i = j - 1
                    break
            i += 1
        while i < len(text):
            for j in range(len(text), i, -1):
                if text[i:j] in self.encode_map:
                    encoded.append(self.encode_map[text[i:j]])
                    i = j - 1
                    break
            i += 1
        return encoded

    def decode(self, encoded):
        return ''.join([self.mapping.get(x, '') for x in encoded])
