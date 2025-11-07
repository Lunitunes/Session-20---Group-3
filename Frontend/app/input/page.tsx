'use client'

import { useState } from "react"
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button"
import { Controller, useForm } from "react-hook-form"
import { zodResolver } from "@hookform/resolvers/zod"
import { z } from "zod"
import { FieldGroup, Field, FieldLabel, FieldError, FieldSeparator, FieldContent,  } from "@/components/ui/field";
import { uploadSchema } from "@/schema/uploadSchema";


export default function InputComponent(){
  const [file, setFile] = useState<File | null>(null);
  const [status, setStatus] = useState("");

  const form = useForm<z.infer<typeof uploadSchema>>({
    defaultValues: {
      analysisName:"",
      file: undefined
    },
    resolver: zodResolver(uploadSchema)
  })

  function onSubmit(data: z.infer<typeof uploadSchema>) {
    // Chuck backend stuff here
  }

  return(
    <div className="grid max-w-sm items-center m-auto w-7xl">
      <h1 className="text-5xl my-5 text-center">Input a CSV to get started</h1>
      <div className="grid gap-2 shadow-lg p-4 rounded-lg bg-card w-full border-2">


        <form onSubmit={form.handleSubmit(onSubmit)}>
          <FieldGroup>
            <Controller 
            control={form.control}
            name="analysisName"
            render={({field, fieldState}) => (
            <Field data-invalid={fieldState}>
              <FieldLabel htmlFor="analysisName">Analysis Name:</FieldLabel>
              <Input {...field} id={field.name} type="text" aria-invalid={fieldState.invalid}/>
              {fieldState.invalid && (
              <FieldError errors={[fieldState.error]}/>
              )}
            </Field>
            )}
            />
            <Controller
            control={form.control}
            name="file"
            render={({field, fieldState}) => (
            <Field data-invalid={fieldState}>
              <FieldLabel htmlFor={field.name}>Upload File:</FieldLabel>
              <Input id={field.name} type="file" aria-invalid={fieldState.invalid} accept=".csv"/>
              <FieldError errors={[fieldState.error]}/>
            </Field>
            )}
            />
          </FieldGroup>
          <FieldSeparator className="my-4"/>
          <FieldContent className="">
            <Button type="submit" className="w-full">Submit</Button>
          </FieldContent>
        </form>


      </div>
    </div>
  )
}