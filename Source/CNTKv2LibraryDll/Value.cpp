//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"

#ifdef _WIN32
#define _SCL_SECURE_NO_WARNINGS
#endif

#include "CNTKLibrary.h"
#include "CompositeFunction.h"
#include "Utils.h"
#include "Value.h"

namespace CNTK
{
    Value::Value(const NDArrayViewPtr& data)
        : Value(data, nullptr)
    {
    }

    Value::Value(const NDArrayViewPtr& data, const NDMaskPtr& mask)
        : m_data(data), m_mask(mask)
    {
        if (mask != nullptr)
        {
            auto dataShape = data->Shape();
            auto maskShape = mask->Shape();

            if (maskShape.Rank() > dataShape.Rank())
                InvalidArgument("The rank (%d) of the mask of a Value object cannot exceed the rank (%d) of the data NDArrayView object", (int)maskShape.Rank(), (int)dataShape.Rank());

            if (dataShape.SubShape(dataShape.Rank() - maskShape.Rank()) != maskShape)
                InvalidArgument("Invalid Value object; the data and mask are incompatible. The trailing dimensions of the data with shape %S do not match the dimensions of the mask with shape %S", AsStringForErrorReporting(dataShape).c_str(), AsStringForErrorReporting(maskShape).c_str());
        }
    }

    static NDMaskPtr CreateMask(const std::vector<size_t>& sequenceLengths, const std::vector<bool>& sequenceStartFlags, const DeviceDescriptor& device)
    {
        size_t numSequences = sequenceLengths.size();

        if (!sequenceStartFlags.empty() && (sequenceStartFlags.size() != numSequences))
            InvalidArgument("Value::Create:: The number of sequence start flags does not match the number of sequences");

        std::vector<bool> actualStarts = sequenceStartFlags;
        if (actualStarts.empty())
            actualStarts.resize(numSequences, true);

        size_t maxSequenceLength = 0;
        for (size_t i = 0; i < numSequences; ++i)
            maxSequenceLength = std::max(maxSequenceLength, sequenceLengths[i]);

        bool needsMask = (std::find(actualStarts.begin(), actualStarts.end(), false) != actualStarts.end());
        needsMask = needsMask || (std::find_if(sequenceLengths.begin(), sequenceLengths.end(), [maxSequenceLength](const size_t& currentSequenceLength) {
            return (currentSequenceLength != maxSequenceLength);
        }) != sequenceLengths.end());

        // If needed, create a mask to account for variability in lengths of specified sequences
        NDMaskPtr deviceValueMask;
        if (needsMask)
        {
            NDShape valueMaskShape = { maxSequenceLength, numSequences };
            deviceValueMask = MakeSharedObject<NDMask>(valueMaskShape, device);
            for (size_t i = 0; i < numSequences; ++i)
            {
                if (actualStarts[i])
                    deviceValueMask->MarkSequenceBegin({ 0, i });
                deviceValueMask->InvalidateSection({ sequenceLengths[i], i }, { NDShape::InferredDimension, 1 });
            }
        }

        return deviceValueMask;
    }

    //
    // Create NDMask for the 'sequences' if the 'sequences' do not have the same length.
    // It returns null if all the 'sequences' have the same length.
    //
    template <typename T>
    static NDMaskPtr CreateMask(size_t numElementsPerSample, const std::vector<std::vector<T>>& sequences, const DeviceDescriptor& device)
    {
        size_t numSequences = sequences.size();
        std::vector<size_t> sequenceLengths(numSequences);
        for (size_t i = 0; i < numSequences; ++i)
            sequenceLengths[i] = sequences[i].size() / numElementsPerSample;

        return CreateMask(sequenceLengths, {}, device);
    }

    template <typename ElementType>
    /*static*/ ValuePtr Value::Create(size_t vocabularySize, const std::vector<std::vector<size_t>>& oneHotSequences, const DeviceDescriptor& device, bool readOnly/* = false*/)
    {
        NDMaskPtr deviceValueMask = CreateMask(1, oneHotSequences, DeviceDescriptor::CPUDevice());
        // If deviceValueMask is null, all the sequences have the same length.
        size_t maxSequenceLength = (deviceValueMask == nullptr) ? oneHotSequences[0].size() : deviceValueMask->Shape()[0];

        size_t numSequences = oneHotSequences.size();
        NDShape sampleShape = { vocabularySize };
        NDShape valueDataShape = sampleShape.AppendShape({ maxSequenceLength, numSequences });
        size_t numCSCCols = valueDataShape.SubShape(1).TotalSize() + 1;
        std::vector<SparseIndexType> colStarts(numCSCCols);
        std::vector<ElementType> nonZeroValues;
        std::vector<SparseIndexType> rowIndices;
        for (size_t i = 0; i < numSequences; ++i)
        {
            size_t currentSequenceLength = oneHotSequences[i].size();
            size_t j = 0;
            for (; j < currentSequenceLength; ++j)
            {
                colStarts[(i * maxSequenceLength) + j] = (SparseIndexType)nonZeroValues.size();
                nonZeroValues.push_back(1);
                if (oneHotSequences[i][j] >= vocabularySize)
                    InvalidArgument("Value::Create: one-hot data exceeds vocabulary size");
                rowIndices.push_back((SparseIndexType)(oneHotSequences[i][j]));
            }

            for (; j < maxSequenceLength; ++j)
                colStarts[(i * maxSequenceLength) + j] = (SparseIndexType)(nonZeroValues.size());
        }

        colStarts[numSequences * maxSequenceLength] = (SparseIndexType)(nonZeroValues.size());
        NDArrayViewPtr deviceValueData = MakeSharedObject<NDArrayView>(valueDataShape, colStarts.data(), rowIndices.data(), nonZeroValues.data(), nonZeroValues.size(), device, readOnly);
        return MakeSharedObject<Value>(deviceValueData, deviceValueMask);
    }

    /*static*/ ValuePtr Value::Create(const NDShape& sampleShape, const std::vector<NDArrayViewPtr>& sequences, const std::vector<bool>& sequenceStartFlags, const DeviceDescriptor& device, bool readOnly, bool createNewCopy)
    {
        auto numSequences = sequences.size();
        if (numSequences == 0)
            InvalidArgument("Value::Create:: The number of sequences is 0");

        std::vector<size_t> sequenceLengths(numSequences);
        size_t maxSequenceLength = 0;
        DataType dataType = sequences[0]->GetDataType();
        for (size_t i = 0; i < numSequences; ++i)
        {
            auto currentSequenceData = sequences[i];
            if (currentSequenceData->GetDataType() != dataType)
                InvalidArgument("Value::Create:: All NDArrayView objects should have the same data type");

            if ((numSequences > 1) && (currentSequenceData->Device() != DeviceDescriptor::CPUDevice()))
                InvalidArgument("Value::Create:: All NDArrayView objects should be located on the CPU");

            auto currentSequenceDataShape = currentSequenceData->Shape();
            if ((currentSequenceDataShape.Rank() < sampleShape.Rank()) || (currentSequenceDataShape.Rank() > (sampleShape.Rank() + 1)) || (currentSequenceDataShape.SubShape(0, sampleShape.Rank()) != sampleShape))
                InvalidArgument("Value::Create:: The shape of the sequence %lu (%S) is not compatible with the sample shape (%S)", (unsigned long)i, AsStringForErrorReporting(currentSequenceDataShape).c_str(), AsStringForErrorReporting(sampleShape).c_str());

            sequenceLengths[i] = currentSequenceDataShape.SubShape(sampleShape.Rank()).TotalSize();
            maxSequenceLength = std::max(maxSequenceLength, sequenceLengths[i]);
        }

        NDMaskPtr deviceValueMask = CreateMask(sequenceLengths, sequenceStartFlags, DeviceDescriptor::CPUDevice());

        NDArrayViewPtr valueData;
        if (numSequences == 1)
        {
            if (createNewCopy)
                valueData = sequences[0]->DeepClone();
            else
                valueData = sequences[0];
        }
        else
        {
            NDShape valueDataShape = sampleShape.AppendShape({ maxSequenceLength, numSequences });
            valueData = MakeSharedObject<NDArrayView>(dataType, valueDataShape, DeviceDescriptor::CPUDevice());
            auto maxSequenceSizeInElements = sampleShape.TotalSize() * maxSequenceLength;
            switch (dataType)
            {
            case DataType::Float:
            {
                float* dataBuffer = valueData->WritableDataBuffer<float>();
                for (size_t i = 0; i < numSequences; ++i)
                {
                    const float* currentSequenceBuffer = sequences[i]->DataBuffer<float>();
                    auto currentSequenceSizeInElements = sequences[i]->Shape().TotalSize();
                    std::copy(currentSequenceBuffer, currentSequenceBuffer + currentSequenceSizeInElements, dataBuffer + (maxSequenceSizeInElements * i));
                }
                break;
            }
            case DataType::Double:
            {
                double* dataBuffer = valueData->WritableDataBuffer<double>();
                for (size_t i = 0; i < numSequences; ++i)
                {
                    const double* currentSequenceBuffer = sequences[i]->DataBuffer<double>();
                    auto currentSequenceSizeInElements = sequences[i]->Shape().TotalSize() ;
                    std::copy(currentSequenceBuffer, currentSequenceBuffer + currentSequenceSizeInElements, dataBuffer + (maxSequenceSizeInElements * i));
                }
                break;
            }
            default:
                NOT_IMPLEMENTED;
            }
        }

        NDArrayViewPtr deviceValueData;
        if (device == DeviceDescriptor::CPUDevice())
        {
            if (readOnly)
                deviceValueData = valueData->Alias(readOnly);
            else
                deviceValueData = valueData;
        }
        else
        deviceValueData = valueData->DeepClone(device, readOnly);

        return MakeSharedObject<Value>(deviceValueData, deviceValueMask);
    }

    template <typename ElementType>
    /*static*/ ValuePtr Value::Create(const NDShape& sampleShape, const std::vector<std::vector<ElementType>>& sequences, const std::vector<bool>& sequenceStartFlags, const DeviceDescriptor& device, bool readOnly)
    {
        // Create a NDArrayView object wrapping each of the vectors representing a sequence 
        size_t numElementsPerSample = sampleShape.TotalSize();
        size_t numSequences = sequences.size();
        std::vector<NDArrayViewPtr> sequencesData;
        for (size_t i = 0; i < numSequences; ++i)
        {
            auto& currentSequence = sequences[i];
            if ((currentSequence.size() % numElementsPerSample) != 0)
                InvalidArgument("Value::Create:: The number of elements in the vector containing sequence data must be a multiple of the size of specified sampel shape");

            auto sequenceLength = currentSequence.size() / numElementsPerSample;
            auto sequenceDataShape = sampleShape.AppendShape({sequenceLength});
            sequencesData.push_back(MakeSharedObject<NDArrayView>(sequenceDataShape, currentSequence));
        }

        return Create(sampleShape, sequencesData, sequenceStartFlags, device, readOnly, /*createNewCopy =*/ true);
    }

    /*virtual*/ Value::~Value()
    {
    }

    /*virtual*/ NDArrayViewPtr Value::Data() const
    {
        // TODO: Check if this is a derived type and throw an exception in that case
        return m_data;
    }

    /*virtual*/ NDMaskPtr Value::Mask() const
    {
        // TODO: Check if this is a derived type and throw an exception in that case
        return m_mask;
    }

    /*virtual*/ ValuePtr Value::DeepClone(bool readOnly/* = false*/) const
    {
        // TODO: Check if this is a derived type and throw an exception in that case
        return MakeSharedObject<Value>(Data()->DeepClone(readOnly), (Mask() != nullptr) ? Mask()->DeepClone() : nullptr);
    }

    /*virtual*/ ValuePtr Value::Alias(bool readOnly/* = false*/) const
    {
        // TODO: Check if this is a derived type and throw an exception in that case
        return MakeSharedObject<Value>(Data()->Alias(readOnly), (Mask() != nullptr) ? Mask()->Alias() : nullptr);
    }

    /*virtual*/ void Value::CopyFrom(const Value& source)
    {
        // TODO: Check if this is a derived type and throw an exception in that case
        Data()->CopyFrom(*source.Data());
        if ((Mask() == nullptr) && (source.Mask() != nullptr))
            InvalidArgument("Value::CopyFrom: Invalid source object; Cannot copy a Value with a mask into 'this' Value that does not have a mask.");

        if (source.Mask() != nullptr)
            Mask()->CopyFrom(*source.Mask());
        else
        {
            if (Mask() != nullptr)
            {
                // Clear the mask
                Mask()->Clear();
            }
        }
    }

    template <typename ElementType, typename DestType>
    void DirectCopy(const ElementType *source, const size_t sampleCount, const size_t sampleSize, std::vector<DestType>& dest, size_t& destSampleStart);

    template <typename ElementType, typename DestType>
    void CopyDenseToOneHot(const ElementType *source, const size_t sampleCount, const size_t sampleSize, std::vector<DestType>& dest, size_t& destSampleStart);

    template <typename ElementType>
    void Value::CopyToVector(const NDShape& sampleShape, std::vector<std::vector<ElementType>>& sequences, std::vector<size_t>& sequenceLens)
    { 
        // Check the data type matches
        if (AsDataType<ElementType>() != GetDataType())
            InvalidArgument("The specified ElementType %s does not match the DataType %s", typeid(ElementType).name(), DataTypeName(GetDataType()));

        // Todo: convert sparse into dense.
        if (GetStorageFormat() != StorageFormat::Dense)
            InvalidArgument("Only the dense storage format is supported now.");

        //std::function<void(const float*, const size_t, const size_t, std::vector<float>&, size_t&)> myCopy = Value::DirectCopy<float>;

        CopyToImpl<ElementType, ElementType>(sampleShape, sequences, sequenceLens);
    }

    template <typename ElementType>
    CNTK_API void Value::CopyToVector(const NDShape& sampleShape, std::vector<std::vector<size_t>>& sequences, std::vector<size_t>& sequenceLens)
    {
        // Todo: convert sparse into dense.
        if (GetStorageFormat() != StorageFormat::Dense)
            InvalidArgument("Only the dense storage format is supported now.");

        if (sampleShape[0] != sampleShape.TotalSize())
            InvalidArgument("");

        CopyToImpl<ElementType, size_t>(sampleShape, sequences, sequenceLens);
    }

    template <typename ValueType, typename DestType>
    void Value::CopyToImpl(const NDShape& sampleShape, 
                           std::vector<std::vector<DestType>>& sequences, 
                           std::vector<size_t>& sequenceLens)
    {
        auto valueRank = Shape().Rank();
        auto sampleRank = sampleShape.Rank();
        if ((valueRank < sampleRank + 1) || (valueRank > sampleRank + 2) || (sampleShape != Shape().SubShape(0, sampleRank)))
            RuntimeError("The variable and the Value does not have the same tensor shape.");

        size_t numOfSequences;
        size_t maxSequenceLen;
        if (valueRank == sampleShape.Rank() + 1)
        {
            // no batch axis, only sequence axis
            numOfSequences = 1;
            maxSequenceLen = Shape()[valueRank - 1];
        }
        else
        {
            assert(valueRank == sampleShape.Rank() + 2);
            numOfSequences = Shape()[valueRank - 1];
            maxSequenceLen = Shape()[valueRank - 2];
        }

        // Check output buffer size
        if (sequences.size() < numOfSequences)
            RuntimeError("The size of output buffer is too small");

        // Check sequenceLens size.
        if (sequenceLens.size() < numOfSequences)
        {
            RuntimeError("The size of sequenceLens does not match.");
        }
        else
        {
            for (size_t i = numOfSequences; i < sequenceLens.size(); i++)
                sequenceLens[i] = 0;
        }

        // Copy data to the CPU device if required.
        const ValueType *valueData;
        const MaskKind* maskData = nullptr;
        if (Device() != DeviceDescriptor::CPUDevice())
        {
            valueData = m_data->DeepClone(DeviceDescriptor::CPUDevice())->DataBuffer<ValueType>();
            if (m_mask != nullptr)
                maskData = m_mask->DeepClone(DeviceDescriptor::CPUDevice())->DataBuffer();
        }
        else
        {
            valueData = m_data->DataBuffer<ValueType>();
            if (m_mask != nullptr)
                maskData = m_mask->DataBuffer();
        }

        auto sampleSize = sampleShape.TotalSize();
        for (auto seqIndex = 0; seqIndex < numOfSequences; seqIndex++)
        {
            size_t seqStart = seqIndex * maxSequenceLen;
            size_t destSampleCount = 0;
            if (maskData == nullptr)
            {
                // Todo: if function pointer or lambda could support template, switch to use them.
                if (typeid(DestType) == typeid(size_t))
                {
                    CopyDenseToOneHot<ValueType, DestType>(valueData + seqStart * sampleSize, maxSequenceLen, sampleSize, sequences[seqIndex], destSampleCount);
                }
                else
                {
                    DirectCopy<ValueType, DestType>(valueData + seqStart * sampleSize, maxSequenceLen, sampleSize, sequences[seqIndex], destSampleCount);
                }
                sequenceLens[seqIndex] = destSampleCount;
            }
            else
            {
                // NDMask is not null
                size_t current = seqStart;
                size_t seqEnd = seqStart + maxSequenceLen;
                while (current < seqEnd)
                {
                    // find first valid mask.
                    while ((current < seqEnd) && (maskData[current] == MaskKind::Invalid))
                        current++;
                    auto sampleStart = current;

                    // find the next invalid mask.
                    while ((current < seqEnd) && (maskData[current] != MaskKind::Invalid))
                        current++;
                    assert(current >= sampleStart);
                    if (current > sampleStart)
                    {
                        // Todo: if function pointer or lambda could support template, switch to use them.
                        if (typeid(DestType) == typeid(size_t))
                        {
                            CopyDenseToOneHot<ValueType, DestType>(valueData + seqStart * sampleSize, current - sampleStart, sampleSize, sequences[seqIndex], destSampleCount);
                        }
                        else
                        {
                            DirectCopy<ValueType, DestType>(valueData + seqStart * sampleSize, current - sampleStart, sampleSize, sequences[seqIndex], destSampleCount);
                        }
                    }
                }
                sequenceLens[seqIndex] = destSampleCount;
            }
        }
    }

    void PackedValue::Unpack() const
    {
        if (m_packedDataLayout && (m_packedDataLayout->GetNumTimeSteps() != 1) && (m_packedDataLayout->GetNumSequences() != 1) && Internal::IsAutomaticUnpackingOfPackedValuesDisabled())
            LogicError("PackedValue::Unpack: Automatic unpacking of PackedValue objects is disabled");

        if (m_isPacked)
        {
            ValuePtr valueObject;
            auto dataType = m_packedData->GetDataType();
            switch (dataType)
            {
            case DataType::Float:
                valueObject = Utils::GetValueObjectFromCNTKImplMatrixAndMBLayout(m_sampleShape, *(m_packedData->GetMatrix<float>()), m_packedDataLayout, m_isReadOnly);
                break;
            case DataType::Double:
                valueObject = Utils::GetValueObjectFromCNTKImplMatrixAndMBLayout(m_sampleShape, *(m_packedData->GetMatrix<double>()), m_packedDataLayout, m_isReadOnly);
                break;
            default:
                LogicError("Unsupported DataType %s", DataTypeName(dataType));
            }

            m_data = valueObject->Data();
            m_mask = valueObject->Mask();

            m_packedData = nullptr;
            m_packedDataLayout = nullptr;
            m_isPacked = false;

            if (m_unpackedShape != m_data->Shape())
                LogicError("The computed unpacked shape of the PackedValue object does not match the actual Data NDArrayView's shape after unpacking");
        }
    }

    template <typename ElementType, typename DestType>
    void DirectCopy(const ElementType *source, const size_t sampleCount, const size_t sampleSize, std::vector<DestType>& dest, size_t& destSampleStart)
    {
        if (typeid(ElementType) != typeid(DestType))
            RuntimeError("Source and destination must be the same data type.");

        DestType *destData = dest.data();
        if ((destSampleStart + sampleCount) * sampleSize > dest.size())
            RuntimeError("The output buffer is too small.");
        std::copy(source, source + sampleCount * sampleSize, reinterpret_cast<ElementType *>(destData + destSampleStart * sampleSize));
        destSampleStart += sampleCount;
    }

    template <typename ElementType, typename DestType>
    void CopyDenseToOneHot(const ElementType *source, const size_t sampleCount, const size_t sampleSize, std::vector<DestType>& dest, size_t& destSampleStart)
    {
        if (typeid(DestType) != typeid(size_t))
        {
            RuntimeError("The destination data type must be size_t.");
        }

        const ElementType *currentp = source;
        const ElementType *lastp = source + sampleCount * sampleSize;
        while (currentp < lastp)
        {
            auto sampleEndp = currentp + sampleSize;
            auto indexp = std::find_if(currentp, sampleEndp, [](const ElementType val) {
                return val != 0;
            });

            if (indexp == sampleEndp)
            {
                RuntimeError("Cannot convert to onehot vector: the sample does not have any non-zero value.");
            }
            else
            {
                if (std::find_if(indexp + 1, sampleEndp, [](const ElementType val) {
                    return val != 0;
                }) != sampleEndp)
                {
                    RuntimeError("Cannot convert to onehot vector: more than one non-zero value in the sample.");
                }
                else
                {
                    if (destSampleStart >= dest.size())
                        RuntimeError("The output buffer is too small.");
                    else
                    {
                        dest[destSampleStart++] = static_cast<DestType>(indexp - currentp);
                    }
                }
            }
            currentp += sampleSize;
        }
        assert(currentp == lastp);
    }

    // Explicit template instantiations
    template /*static*/ CNTK_API ValuePtr Value::Create<float>(const NDShape& sampleShape, const std::vector<std::vector<float>>& sequences, const std::vector<bool>& sequenceStartFlags, const DeviceDescriptor& device, bool readOnly/* = false*/);
    template /*static*/ CNTK_API ValuePtr Value::Create<double>(const NDShape& sampleShape, const std::vector<std::vector<double>>& sequences, const std::vector<bool>& sequenceStartFlags, const DeviceDescriptor& device, bool readOnly/* = false*/);
    template /*static*/ CNTK_API ValuePtr Value::Create<float>(size_t vocabSize, const std::vector<std::vector<size_t>>& oneHotSequences, const DeviceDescriptor& device, bool readOnly/* = false*/);
    template /*static*/ CNTK_API ValuePtr Value::Create<double>(size_t vocabSize, const std::vector<std::vector<size_t>>& oneHotSequences, const DeviceDescriptor& device, bool readOnly/* = false*/);
    template CNTK_API void Value::CopyToVector<float>(const NDShape& sampleShape, std::vector<std::vector<float>>& sequences, std::vector<size_t>& sequencesLens);
    template CNTK_API void Value::CopyToVector<double>(const NDShape& sampleShape, std::vector<std::vector<double>>& sequences, std::vector<size_t>& sequencesLens);

}
