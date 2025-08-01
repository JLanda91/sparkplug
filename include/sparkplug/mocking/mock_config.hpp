#pragma once

#include <vector>
#include <stdexcept>

namespace sparkplug::mocking {

    template<typename Signature>
    class MockConfig {
    public:
        MockConfig(const std::vector<typename Signature::input_t>& expected_inputs, const std::vector<typename Signature::return_t>& mocked_outputs)
            : expected_inputs_(expected_inputs), mocked_outputs_(mocked_outputs)
        {
            if (expected_inputs.size() != mocked_outputs.size()) {
                throw std::invalid_argument("expected inputs and outputs have different size");
            }
        }

        [[nodiscard]] auto expected_inputs() const noexcept -> const std::vector<int>& {
            return expected_inputs_;
        }

        [[nodiscard]] auto mocked_outputs() const noexcept -> const std::vector<int>& {
            return mocked_outputs_;
        }

        [[nodiscard]] auto size() const noexcept {
            return expected_inputs_.size();
        }

    private:
        std::vector<typename Signature::input_t> expected_inputs_;
        std::vector<typename Signature::return_t> mocked_outputs_;
    };

}
